"""Score your checkpoints side by side (Chapters 7, 14 and 15).

  python eval.py                                 # every checkpoint it finds
  python eval.py --ckpts chat.pt chat_dpo.pt --n 100

Two measurements:
  perplexity  on held-out stories (val.bin): how well the model predicts real text. Lower is better.
  following   for chat models, on prompt phrasings never used in training:
                topic word  how often the story contains the requested word. A word match only:
                            it says nothing about whether the story is good or coherent.
                finished    how often the model ends the story itself (emits <|endoftext|>)
                repeats     share of 4-word phrases that repeat earlier ones; high means looping
                length      average tokens per story
Read the numbers together and read the samples: no single one of them measures story quality.
Every model gets the same prompts and the same random seeds, so differences come from the weights.
"""
import argparse, json, math, os, random, re, numpy as np, torch
from model import GPT, GPTConfig
from bpe import BPETokenizer
from device import pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpts", nargs="*", default=None)
p.add_argument("--n", type=int, default=50, help="prompts per chat model")
p.add_argument("--max_new", type=int, default=200)
p.add_argument("--json", default=None, help="also write results and sample stories here")
args = p.parse_args()

device = pick_device()
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
ckpts = args.ckpts or [c for c in ["ckpt.pt", "chat.pt", "chat_dpo.pt", "chat_ppo.pt"] if os.path.exists(c)]

TOPICS = ["dog", "cat", "ball", "tree", "bird", "fish", "car", "cake", "boat", "flower",
          "dragon", "star", "moon", "bear", "frog", "train", "kite", "rabbit", "princess", "robot"]
TEMPLATES = ["Please write a story where a {w} is the hero.",          # none of these phrasings
             "Could you tell me a short tale about a {w}?",            # appear in make_sft.py
             "Make up a bedtime story featuring a {w}."]
rng = random.Random(0)
PROMPTS = [(w, rng.choice(TEMPLATES).format(w=w)) for w in (rng.choice(TOPICS) for _ in range(args.n))]

def load(path):
    ck = torch.load(path, map_location=device)
    m = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    return m

@torch.no_grad()
def perplexity(m, batches=20, B=16):
    val = np.memmap("val.bin", dtype=np.uint16, mode="r")
    T = min(256, m.cfg.block_size)
    g = np.random.default_rng(0)                       # the same text for every model
    tot = 0.0
    for _ in range(batches):
        ix = g.integers(0, len(val) - T - 1, B)
        x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(val[i + 1:i + 1 + T].astype(np.int64)) for i in ix]).to(device)
        tot += m(x, y)[1].item()
    return math.exp(tot / batches)

@torch.no_grad()
def following(m):
    on_topic = finished = length = 0
    rep, samples = [], []
    for i, (word, prompt) in enumerate(PROMPTS):
        torch.manual_seed(i)                           # same sampling noise for every model
        ids = tok.encode(f"<|user|>{prompt}<|assistant|>")
        out = m.generate_cached(torch.tensor([ids], device=device), args.max_new, temperature=0.8, top_k=50)
        resp = out[0, len(ids):].tolist()
        if EOT in resp:
            finished += 1
            resp = resp[:resp.index(EOT)]
        text = tok.decode([t for t in resp if t < EOT])
        hit = bool(re.search(rf"\b{word}s?\b", text.lower()))
        on_topic += hit
        length += len(resp)
        words = re.findall(r"[a-z']+", text.lower())
        grams = [tuple(words[j:j + 4]) for j in range(len(words) - 3)]
        rep.append(1 - len(set(grams)) / len(grams) if grams else 0.0)
        samples.append({"seed": i, "prompt": prompt, "topic": word, "topic_word_present": hit,
                        "finished": EOT in out[0, len(ids):].tolist(), "story": text})
    n = len(PROMPTS)
    return {"topic_word": on_topic / n, "finished": finished / n, "repeats": sum(rep) / n,
            "length": length / n, "samples": samples}

results = {"prompts": len(PROMPTS), "max_new": args.max_new, "checkpoints": {}}
print(f"{'checkpoint':14s} {'perplexity':>10s} {'topic word':>11s} {'finished':>9s} {'repeats':>8s} {'length':>7s}")
for path in ckpts:
    m = load(path)
    row = {"perplexity": perplexity(m)}
    if path.startswith("chat"):
        row.update(following(m))
        print(f"{path:14s} {row['perplexity']:10.2f} {row['topic_word']:11.0%} {row['finished']:9.0%} "
              f"{row['repeats']:8.0%} {row['length']:7.0f}")
    else:
        print(f"{path:14s} {row['perplexity']:10.2f} {'':>11s} {'':>9s} {'':>8s} {'':>7s}   (base model: not trained to follow requests)")
    results["checkpoints"][path] = row
print(f"{len(PROMPTS)} prompts per chat model, same prompts and seeds for each.")
if args.json:
    json.dump(results, open(args.json, "w"), indent=1)
