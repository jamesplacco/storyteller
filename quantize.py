import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from model import GPT, GPTConfig

class QLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0   # one scale per row
        self.register_buffer("wq", torch.round(w / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale)
        self.bias = lin.bias
    def forward(self, x):
        return F.linear(x, (self.wq.float() * self.scale).to(x.dtype), self.bias)

def quantize_(module, skip=("lm_head",)):          # lm_head is tied to the embedding
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name not in skip:
            setattr(module, name, QLinear(child))
        else:
            quantize_(child, skip)
    return module

val = np.memmap("val.bin", dtype=np.uint16, mode="r")
def val_loss(m, n=20, B=16, T=256):
    g = np.random.default_rng(0)                   # same batches for both models
    tot = 0.0
    with torch.no_grad():
        for _ in range(n):
            ix = g.integers(0, len(val) - T - 1, B)
            x = torch.stack([torch.from_numpy(val[i:i + T].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(val[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
            tot += m(x, y)[1].item()
    return tot / n

ck = torch.load("ckpt.pt", map_location="cpu")
model = GPT(GPTConfig(**ck["cfg"])).eval()
model.load_state_dict(ck["model"])
size = lambda m: sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))

fp_size, fp_loss = size(model), val_loss(model)
quantize_(model)
q_size, q_loss = size(model), val_loss(model)
print(f"fp32: {fp_size / 1e6:5.1f} MB  val {fp_loss:.4f}")
print(f"int8: {q_size / 1e6:5.1f} MB  val {q_loss:.4f}")