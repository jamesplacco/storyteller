import argparse, copy, json, random, torch, torch.nn.functional as F
from model import GPT, GPTConfig
from bpe import BPETokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
random.seed(0)
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=300)
args = p.parse_args()
T, beta, steps, pairs_per_batch = 256, 0.1, args.steps, 8

ck = torch.load("chat.pt", map_location=device)
policy = GPT(GPTConfig(**ck["cfg"])).to(device)
T = min(T, policy.cfg.block_size)                  # never longer than the model's context
policy.load_state_dict(ck["model"])
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad = False

data = json.load(open("sft.json"))
def encode(prompt, resp):
    p = tok.encode(f"<|user|>{prompt}<|assistant|>")
    r = tok.encode_ordinary(resp) + [EOT]
    ids, lab = (p + r)[:T + 1], ([-100] * len(p) + r)[:T + 1]
    x, y = ids[:-1], lab[1:]
    return x + [EOT] * (T - len(x)), y + [-100] * (T - len(y))

def batch():
    xs, ys = [], []
    for ex in random.sample(data, pairs_per_batch):
        for _ in range(100):                               # find a story about something else
            bad = random.choice(data)
            if ex["word"] not in bad["response"].lower():
                break
        for resp in (ex["response"], bad["response"]):     # chosen, rejected
            x, y = encode(ex["prompt"], resp); xs.append(x); ys.append(y)
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)

def seq_logp(model, x, y):
    logp = F.log_softmax(model(x)[0], dim=-1)
    tok_lp = logp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_lp * (y != -100)).sum(-1)                 # sum over response tokens

opt = torch.optim.AdamW(policy.parameters(), lr=5e-6)
for step in range(steps):
    x, y = batch()
    pi = seq_logp(policy, x, y)
    with torch.no_grad():
        rf = seq_logp(ref, x, y)
    margin = beta * ((pi[0::2] - rf[0::2]) - (pi[1::2] - rf[1::2]))
    loss = -F.logsigmoid(margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 25 == 0:
        print(f"step {step:3d} | loss {loss.item():.3f} | pref acc {(margin > 0).float().mean().item():.2f}")

torch.save({"model": policy.state_dict(), "cfg": ck["cfg"]}, "chat_dpo.pt")