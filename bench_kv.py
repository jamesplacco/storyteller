import time, torch
from model import GPT, GPTConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"
ck = torch.load("ckpt.pt", map_location=device)
model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
model.load_state_dict(ck["model"])
idx = torch.zeros((1, 8), dtype=torch.long, device=device)

for name, fn in [("no cache", model.generate), ("kv-cache", model.generate_cached)]:
    torch.manual_seed(0); t = time.time()
    out = fn(idx, 240, top_k=50)
    if device == "mps": torch.mps.synchronize()
    n = out.size(1) - idx.size(1)                 # the cached version stops at the context limit
    print(f"{name:9s} {n / (time.time() - t):6.0f} tok/s  ({n} tokens)")