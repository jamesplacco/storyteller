import argparse, time, json, random, torch, torch.nn as nn
from model import GPT, GPTConfig
from bpe import BPETokenizer
from device import pick_device

device = pick_device()
random.seed(0); torch.manual_seed(0)
tok = BPETokenizer().load("tok.pkl")
EOT = tok.special["<|endoftext|>"]
p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1500)
args = p.parse_args()
B, T, steps = 16, 256, args.steps

class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: start at the base model
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

ck = torch.load("ckpt.pt", map_location=device)
model = GPT(GPTConfig(**ck["cfg"])).to(device)
T = min(T, model.cfg.block_size)                   # never longer than the model's context
model.load_state_dict(ck["model"])
for p in model.parameters():
    p.requires_grad = False
# The chat tokens never appeared in pretraining, so their embedding rows are untrained. Learn
# just those rows: a gradient mask zeroes every other row of the (tied) embedding.
model.wte.weight.requires_grad = True
chat_rows = torch.zeros(model.cfg.vocab_size, 1, device=device)
chat_rows[list(tok.special.values())] = 1.0
model.wte.weight.register_hook(lambda g: g * chat_rows)
for blk in model.blocks:
    blk.attn.qkv = LoRALinear(blk.attn.qkv).to(device)
    blk.attn.proj = LoRALinear(blk.attn.proj).to(device)
trainable = [p for p in model.parameters() if p.requires_grad]
n_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n != "wte.weight")
n_emb = len(tok.special) * model.cfg.n_embd
print(f"trainable {n_lora + n_emb:,} of {sum(p.numel() for p in model.parameters()):,} "
      f"({n_lora:,} in LoRA adapters, {n_emb:,} in the chat-token embeddings)")

data = json.load(open("sft.json"))
def encode(ex):
    p = tok.encode(f"<|user|>{ex['prompt']}<|assistant|>")
    r = tok.encode_ordinary(ex["response"]) + [EOT]
    ids = (p + r)[:T + 1]
    lab = ([-100] * len(p) + r)[:T + 1]              # no loss on the prompt
    x, y = ids[:-1], lab[1:]
    return x + [EOT] * (T - len(x)), y + [-100] * (T - len(y))

def get_batch():
    xs, ys = zip(*(encode(ex) for ex in random.sample(data, B)))
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)

opt = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
t_start = time.time()
for step in range(steps):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.3f}")

with torch.no_grad():                                   # merge adapters into W
    for blk in model.blocks:
        for name in ("qkv", "proj"):
            l = getattr(blk.attn, name)
            l.base.weight += l.scale * (l.B @ l.A)
            setattr(blk.attn, name, l.base)
torch.save({"model": model.state_dict(), "cfg": ck["cfg"],
            "train": {**vars(args), "script": "sft.py", "from": "ckpt.pt", "seconds": round(time.time() - t_start, 1)}}, "chat.pt")

model.eval()
idx = torch.tensor([tok.encode("<|user|>Tell me a bedtime story with a dragon in it.<|assistant|>")], device=device)
out = model.generate(idx, 200, temperature=0.8, top_k=50)[0].tolist()
print(tok.decode(out).split("<|assistant|>")[1].split("<|endoftext|>")[0])