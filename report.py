"""Write an experiment report from the checkpoints in this folder.

  python report.py                    # -> reports/<date>-<commit>/REPORT.md, manifest.json, images
  python report.py --name first-run --n 50

Run it after training on real data. It records what was run and what came out, with nothing
picked by hand:
  - the exact code version, library versions and hardware
  - SHA-256 checksums of the tokenizer, data files and every checkpoint
  - each checkpoint's architecture, parameter count and recorded training settings
  - eval.py's table, plus every sample story it generated (fixed prompts and seeds, failures included)
  - illustrations from the image models, if they exist
  - C engine agreement, KV-cache speed and int8 quantization, measured on this machine
Rerun it and you get the same prompts and seeds; the stories differ only if the weights do.
"""
import argparse, datetime, hashlib, json, os, platform, re, shutil, statistics, subprocess, sys, time
import torch
from device import pick_device, synchronize

p = argparse.ArgumentParser()
p.add_argument("--name", default="")
p.add_argument("--n", type=int, default=50, help="prompts per chat model in the evaluation")
p.add_argument("--show", type=int, default=6, help="sample stories per model written into the report")
p.add_argument("--showcase", default="chat.pt", help="chat model whose story goes at the top of README.md")
args = p.parse_args()

def sh(*cmd, timeout=3600):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def mb(path):
    return round(os.path.getsize(path) / 1e6, 2)

# ---------------- where, what, on which machine ----------------
rc, commit, _ = sh("git", "rev-parse", "HEAD")
commit = commit.strip() if rc == 0 else "unknown"
dirty = bool(sh("git", "status", "--porcelain", "--untracked-files=no")[1].strip()) if rc == 0 else None
stamp = datetime.date.today().isoformat()
out = os.path.join("reports", "-".join(x for x in [stamp, commit[:7], args.name] if x))
os.makedirs(out, exist_ok=True)

device = pick_device()
hw = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "device": device}
if sys.platform == "darwin":
    hw["chip"] = sh("sysctl", "-n", "machdep.cpu.brand_string")[1].strip()
    mem = sh("sysctl", "-n", "hw.memsize")[1].strip()
    hw["memory_gb"] = round(int(mem) / 2**30) if mem.isdigit() else None
if torch.cuda.is_available():
    hw["gpu"] = torch.cuda.get_device_name(0)
versions = {"torch": torch.__version__}
for mod in ["numpy", "datasets", "fastapi", "PIL"]:
    try:
        versions[mod] = __import__(mod).__version__
    except Exception:
        versions[mod] = None

files = {f: {"sha256": sha256(f), "mb": mb(f)} for f in ["tok.pkl", "train.bin", "val.bin", "sft.json"] if os.path.exists(f)}
data = {k: os.path.getsize(f) // 2 for k, f in [("train_tokens", "train.bin"), ("val_tokens", "val.bin")] if os.path.exists(f)}
if os.path.exists("sft.json"):
    data["sft_pairs"] = len(json.load(open("sft.json")))

CKPTS = ["ckpt.pt", "chat.pt", "chat_dpo.pt", "chat_ppo.pt", "reward.pt", "vqvae.pt", "image_gpt.pt", "dit.pt"]
models = {}
for f in CKPTS:
    if not os.path.exists(f):
        continue
    ck = torch.load(f, map_location="cpu")
    sd = ck.get("model", ck) if isinstance(ck, dict) else ck
    models[f] = {"sha256": sha256(f), "mb": mb(f),
                 "params": sum(v.numel() for v in sd.values() if hasattr(v, "numel")),
                 "cfg": ck.get("cfg") if isinstance(ck, dict) else None,
                 "train": ck.get("train") if isinstance(ck, dict) else None}
    files[f] = {"sha256": models[f]["sha256"], "mb": models[f]["mb"]}
if "ckpt.pt" not in models:
    raise SystemExit("No ckpt.pt here. Train first (Chapters 6 and 7), then run report.py in the same folder.")

# ---------------- evaluation ----------------
print("running eval.py ...", flush=True)
rc, eval_out, eval_err = sh(sys.executable, "eval.py", "--n", str(args.n), "--json", os.path.join(out, "eval.json"))
ev = json.load(open(os.path.join(out, "eval.json"))) if rc == 0 else None
eval_table = "\n".join(l for l in eval_out.splitlines() if l.strip())

# ---------------- base model samples ----------------
from model import GPT, GPTConfig
from bpe import BPETokenizer
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
def load(f):
    ck = torch.load(f, map_location=device)
    m = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
    m.load_state_dict(ck["model"])
    return m
def story(m, prompt, n=180, seed=0, greedy=False):
    torch.manual_seed(seed)
    ids = tok.encode(prompt)
    o = m.generate_cached(torch.tensor([ids], device=device), n, temperature=1.0 if greedy else 0.8, top_k=1 if greedy else 50)
    gen = o[0, len(ids):].tolist()
    gen = gen[:gen.index(EOT)] if EOT in gen else gen
    return tok.decode([t for t in gen if t < EOT])
base = load("ckpt.pt")
with torch.no_grad():
    base_samples = [("greedy", story(base, "Once upon a time", greedy=True))] + \
                   [(f"seed {s}", story(base, "Once upon a time", seed=s)) for s in range(3)]

# ---------------- systems measurements ----------------
systems = {}
with torch.no_grad():
    idx = torch.tensor([tok.encode("Once upon a time")], device=device)
    for name, fn in [("no cache", base.generate), ("kv-cache", base.generate_cached)]:
        fn(idx, 16, top_k=1); synchronize(device)                        # warm up
        runs = []
        for _ in range(3):
            t = time.time(); o = fn(idx, 200, top_k=1); synchronize(device)
            runs.append((o.size(1) - idx.size(1)) / (time.time() - t))
        systems[name + " tok/s"] = round(statistics.median(runs))
cc = shutil.which("clang") or shutil.which("gcc")
if cc and sh(sys.executable, "export.py")[0] == 0 and sh(cc, "-O3", "-ffast-math", "-o", "run", "run.c", "-lm")[0] == 0:
    t = time.time(); rc, c_text, c_err = sh("./run", "model.bin", "tok.bin", "-t", "0", "-n", "150", "-i", "Once upon a time")
    with torch.no_grad():
        py_text = "Once upon a time" + story(base, "Once upon a time", n=150, greedy=True)
    systems["C engine matches PyTorch (greedy, 150 tokens)"] = c_text.rstrip("\n") == py_text
    m_c = re.search(r"\((\d+) tok/s\)", c_err)
    systems["C engine tok/s (single CPU thread)"] = int(m_c.group(1)) if m_c else None
rc, q_out, _ = sh(sys.executable, "quantize.py")
for kind, size, loss in re.findall(r"(fp32|int8):\s+([\d.]+) MB\s+val ([\d.]+)", q_out):
    systems[f"{kind} size MB / val loss"] = f"{size} / {loss}"

# ---------------- illustrations ----------------
pictures = []
if os.path.exists("vqvae.pt") and os.path.exists("image_gpt.pt"):
    from vqvae import CLASSES, load_vqvae, to_images, grid
    from image_gpt import load_image_gpt, generate_codes
    vq, ig = load_vqvae("vqvae.pt", device), load_image_gpt("image_gpt.pt", device)
    torch.manual_seed(0)
    with torch.no_grad():
        imgs = [im for c in range(len(CLASSES)) for im in to_images(vq.decode_codes(generate_codes(ig, c, 6)))]
    grid(imgs, ncol=6, scale=4).save(os.path.join(out, "image_gpt.png"))
    pictures.append(("image_gpt.png", "Image GPT (Chapter 17): 6 samples per class, seed 0. Rows: " + ", ".join(CLASSES) + "."))
if os.path.exists("dit.pt"):
    from vqvae import CLASSES, to_images, grid
    from dit import DiT, sample
    d = DiT().to(device); d.load_state_dict(torch.load("dit.pt", map_location=device)); d.eval()
    torch.manual_seed(0)
    y = torch.arange(len(CLASSES), device=device).repeat_interleave(6)
    grid(to_images(sample(d, y)), ncol=6, scale=4).save(os.path.join(out, "dit.png"))
    pictures.append(("dit.png", "Diffusion transformer (Chapter 17): 6 samples per class, seed 0, 50 DDIM steps, guidance 3."))

# ---------------- write it down ----------------
manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit, "uncommitted_changes": dirty,
            "hardware": hw, "versions": versions, "data": data, "files": files, "models": models,
            "eval": {k: v for k, v in (ev or {}).items() if k != "checkpoints"} | {"checkpoints": {
                c: {k: v for k, v in r.items() if k != "samples"} for c, r in (ev or {}).get("checkpoints", {}).items()}},
            "systems": systems}
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1, default=str)

def fmt_train(t):
    if not t:
        return "not recorded"
    keep = ["script", "from", "steps_done", "max_steps", "steps", "iters", "tokens_seen", "batch_size", "block_size",
            "grad_accum", "dtype", "max_lr", "device", "world_size", "seconds", "val_loss"]
    return ", ".join(f"{k} {t[k]:,}" if isinstance(t[k], int) and not isinstance(t[k], bool) else f"{k} {t[k]}" for k in keep if k in t)

L = []
L.append(f"# Storyteller experiment report{': ' + args.name if args.name else ''}\n")
L.append(f"Generated {manifest['created']} by `report.py` at commit `{commit[:7]}`"
         + (" (with uncommitted changes)" if dirty else "") + ". Every number below was measured on this machine; "
         "every sample uses a fixed prompt and seed, and none were picked by hand.\n")
L.append("## Setup\n")
L.append("| | |\n|---|---|")
for k, v in hw.items():
    L.append(f"| {k} | {v} |")
L.append("| libraries | " + ", ".join(f"{k} {v}" for k, v in versions.items() if v) + " |")
for k, v in data.items():
    L.append(f"| {k.replace('_', ' ')} | {v:,} |")
L.append("")
L.append("## Models\n")
L.append("| checkpoint | parameters | MB | training recorded in the checkpoint |\n|---|---:|---:|---|")
for f, m in models.items():
    L.append(f"| `{f}` | {m['params']:,} | {m['mb']} | {fmt_train(m['train'])} |")
L.append("\nSHA-256 checksums for every file are in `manifest.json`.\n")
L.append("## Evaluation\n")
if ev:
    L.append("```\n" + eval_table + "\n```\n")
    L.append("- **perplexity**: on held-out validation stories. Lower is better.\n"
             "- **topic word**: share of stories containing the requested word. A word match only, not a judgment of quality.\n"
             "- **finished**: share of stories the model ended itself. **repeats**: share of 4-word phrases that repeat earlier ones.\n"
             f"- {ev['prompts']} prompts per chat model, phrasings not used in training, the same prompts and seeds for every model.\n")
else:
    L.append("eval.py did not run:\n```\n" + (eval_err[-1500:] or eval_out[-1500:]) + "\n```\n")
L.append("## Sample stories\n")
L.append("### Base model (`ckpt.pt`), prompt \"Once upon a time\"\n")
for label, s in base_samples:
    L.append(f"**{label}:** Once upon a time{s}\n")
if ev:
    chats = [c for c in ev["checkpoints"] if "samples" in ev["checkpoints"][c]]
    for i in range(min(args.show, args.n)):
        first = ev["checkpoints"][chats[0]]["samples"][i] if chats else None
        if not first:
            break
        L.append(f"### Prompt {i + 1}: \"{first['prompt']}\" (seed {first['seed']})\n")
        for c in chats:
            s = ev["checkpoints"][c]["samples"][i]
            flags = ("topic word present" if s["topic_word_present"] else "topic word missing") + ", " + ("finished" if s["finished"] else "cut off")
            L.append(f"**`{c}`** ({flags}): {s['story'].strip() or '(empty)'}\n")
    if chats:
        L.append(f"All {args.n} stories per model are in `eval.json`.\n")
if pictures:
    L.append("## Illustrations\n")
    L.append("The app picks one of CIFAR-10's 10 classes from words in the story and draws a 32×32 picture of that class. "
             "It is conditioned on the class only, not on the story's details.\n")
    for f, cap in pictures:
        L.append(f"![{cap}]({f})\n\n{cap}\n")
L.append("## Systems, measured here\n")
L.append("| measurement | result |\n|---|---|")
for k, v in systems.items():
    L.append(f"| {k} | {v} |")
L.append("")
L.append("## Limitations\n")
L.append("- One training run per stage. No variation across training seeds was measured, so small differences between checkpoints may be noise.\n"
         "- The preference data for the reward model, PPO and DPO is synthetic: the chosen story mentions the requested topic and the "
         "rejected one does not. It teaches the mechanics of preference tuning; it is not human judgment of story quality.\n"
         "- The topic-word metric is a word match. Story quality is not scored automatically; read the samples.\n"
         "- Illustrations are class-conditioned 32×32 images from CIFAR-10, not pictures of the specific story.\n")
L.append("## Reproduce\n")
L.append(f"```bash\ngit checkout {commit[:7]}\npip install -r requirements.txt   # the versions under Setup reproduce this run most closely\n"
         "# then the run order in README.md, and finally:\npython report.py\n```\n"
         "Training on different hardware or library versions will not give identical weights. With these exact checkpoints "
         "(check the SHA-256 in `manifest.json`), greedy decoding is deterministic on a given device.\n")
open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L))

# ---------------- one story and one picture for the top of the README ----------------
SHOW_PROMPT, START, END = "Tell me a bedtime story about a frog.", "<!-- showcase:start -->", "<!-- showcase:end -->"
readme = open("README.md").read() if os.path.exists("README.md") else ""
if os.path.exists(args.showcase) and START in readme and END in readme:
    with torch.no_grad():
        text = story(load(args.showcase), f"<|user|>{SHOW_PROMPT}<|assistant|>", n=200, seed=0).strip()
    block = [START, "## What it makes\n", f"> **\"{SHOW_PROMPT}\"**\n>", "> " + text.replace("\n", "\n> "), ""]
    if os.path.exists("vqvae.pt") and os.path.exists("image_gpt.pt"):
        torch.manual_seed(0)
        with torch.no_grad():
            frog = to_images(vq.decode_codes(generate_codes(ig, CLASSES.index("frog"), 1)))[0]
        frog.resize((192, 192), 0).save(os.path.join(out, "showcase.png"))
        block.append(f'<img src="{out}/showcase.png" width="192" alt="Generated 32x32 picture of a frog">\n')
    block.append(f"Written by `{args.showcase}` ({models[args.showcase]['params'] / 1e6:.1f}M parameters, trained from scratch on this "
                 f"repo's code)" + (", with a class-conditioned 32×32 picture from the image GPT" if len(block) > 5 else "")
                 + f". One fixed prompt, seed 0, not picked from several tries. Full results: [{out}/REPORT.md]({out}/REPORT.md).")
    block.append(END)
    open("README.md", "w").write(readme[:readme.index(START)] + "\n".join(block) + readme[readme.index(END) + len(END):])
    print("updated the showcase at the top of README.md")
print(f"\nwrote {out}/REPORT.md, manifest.json" + ("".join(f", {f}" for f, _ in pictures)))
print(f"Commit it with:  git add {out} README.md && git commit -m 'Experiment report {stamp}' && git push")
