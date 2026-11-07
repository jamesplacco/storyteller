"""Four experiments on number formats (Chapter 9). Runs on CPU in about a minute.

  python precision.py
"""
import torch, torch.nn.functional as F
from model import GPT, GPTConfig

torch.manual_seed(0)
print("1. RANGE: what each format can hold")
for name, dt in [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16),
                 ("fp8 e4m3", torch.float8_e4m3fn), ("fp8 e5m2", torch.float8_e5m2)]:
    fi = torch.finfo(dt)
    print(f"   {name:9s} max {fi.max:<12.4g} smallest normal {fi.tiny:<10.3g} step at 1.0: {fi.eps:.3g}")
g = torch.tensor([1e-8, 3e-7, 70000.0])
print(f"   gradients {g.tolist()} in fp16 -> {g.half().float().tolist()}  (lost, coarse, overflow)")
print(f"   same values in bf16 -> {g.bfloat16().float().tolist()}")

print("\n2. PRECISION: why the master weights stay in fp32")
for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
    w = torch.tensor(1.0, dtype=dt)
    for _ in range(1000):
        w = w + torch.tensor(1e-4, dtype=dt)      # a typical lr * grad update
    print(f"   {name}: 1.0 plus 1,000 updates of 1e-4 = {w.item():.4f}  (should be 1.1000)")

print("\n3. FP8 MATMUL: scaling decides whether fp8 works")
a, b = torch.randn(256, 256) * 0.003, torch.randn(256, 256) * 0.003  # small values, like real activations
ref = a @ b
def fp8_mm(a, b, scaled):
    sa = a.abs().max() / 448 if scaled else torch.tensor(1.0)        # 448 = largest e4m3 value
    sb = b.abs().max() / 448 if scaled else torch.tensor(1.0)
    qa, qb = (a / sa).to(torch.float8_e4m3fn), (b / sb).to(torch.float8_e4m3fn)
    return (qa.float() @ qb.float()) * sa * sb                        # real kernels fuse this
for scaled in (False, True):
    out = fp8_mm(a, b, scaled)
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"   e4m3 {'with' if scaled else 'without'} per-tensor scaling: relative error {rel:.3f}")

print("\n4. TRAINING: 150 steps of a small GPT on a fixed synthetic task, four ways")
cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=64)
data = torch.randint(0, 64, (512, 33))
data[:, 1::2] = (data[:, 0::2][:, :16] * 7 + 3) % 64                  # every odd token is predictable
def train(mode):
    torch.manual_seed(0)
    model = GPT(cfg)
    if mode == "pure bf16":
        model = model.to(torch.bfloat16)                               # weights and Adam state in bf16
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cpu") if mode == "fp16 + scaler" else None
    amp = {"bf16 autocast": torch.bfloat16, "fp16 + scaler": torch.float16}.get(mode)
    for step in range(150):
        batch = data[torch.randint(0, 512, (32,))]
        x, y = batch[:, :-1], batch[:, 1:]
        with torch.autocast("cpu", dtype=amp, enabled=amp is not None):
            _, loss = model(x, y)
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    return loss.float().item()
for mode in ["fp32", "bf16 autocast", "fp16 + scaler", "pure bf16"]:
    print(f"   {mode:14s} final loss {train(mode):.3f}")
print("   Autocast keeps fp32 master weights, so it tracks fp32. Pure bf16 loses small updates (experiment 2).")
