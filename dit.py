"""A diffusion transformer (DiT) on CIFAR-10 (Chapter 17). The other way to make pictures.

No tokens: the model learns to remove Gaussian noise. Training adds a random amount of noise
to a real image and asks the transformer to predict that noise. Sampling starts from pure noise
and removes it step by step (DDIM, 50 steps), steered by the class with classifier-free guidance.

The transformer is Chapter 5's block with three changes: no causal mask, the image cut into
4x4 patches as tokens, and adaLN-Zero (timestep and class set each LayerNorm's scale and shift).

  python dit.py          # -> dit.pt, dit_samples.png
"""
import argparse, math, torch, torch.nn as nn, torch.nn.functional as F
from vqvae import CLASSES, load_cifar, to_images, grid
from device import pick_device

PATCH, C, LAYERS, HEADS, NSTEPS = 4, 256, 6, 8, 1000
N_TOK, P_DIM = (32 // PATCH) ** 2, 3 * PATCH * PATCH          # 64 tokens of 48 numbers each

betas = torch.linspace(1e-4, 0.02, NSTEPS)                     # DDPM noise schedule
alpha_bar = torch.cumprod(1 - betas, 0)                        # fraction of signal left at step t

def patchify(x):       # (B, 3, 32, 32) -> (B, 64, 48)
    B = x.size(0)
    return x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH).permute(0, 2, 3, 1, 4, 5).reshape(B, N_TOK, P_DIM)

def unpatchify(p):     # (B, 64, 48) -> (B, 3, 32, 32)
    B, g = p.size(0), 32 // PATCH
    return p.reshape(B, g, g, 3, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 32, 32)

def timestep_embedding(t, dim=C):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(C, 3 * C), nn.Linear(C, C)
        self.ln2 = nn.LayerNorm(C, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(approximate="tanh"), nn.Linear(4 * C, C))
        self.ada = nn.Linear(C, 6 * C)                    # condition -> 2 shifts, 2 scales, 2 gates
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)   # "Zero": block starts as identity

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(F.silu(c)).chunk(6, -1)
        B, T, _ = x.shape
        q, k, v = self.qkv(modulate(self.ln1(x), s1, sc1)).split(C, -1)
        q, k, v = [t.view(B, T, HEADS, C // HEADS).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v)       # every patch sees every patch
        x = x + g1.unsqueeze(1) * self.proj(a.transpose(1, 2).reshape(B, T, C))
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.ln2(x), s2, sc2))
        return x

class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_in = nn.Linear(P_DIM, C)
        self.pos = nn.Parameter(torch.randn(1, N_TOK, C) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(C, C), nn.SiLU(), nn.Linear(C, C))
        self.y_emb = nn.Embedding(len(CLASSES) + 1, C)    # last index = "no class", for guidance
        self.blocks = nn.ModuleList([DiTBlock() for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(C, elementwise_affine=False)
        self.ada_f = nn.Linear(C, 2 * C)
        self.patch_out = nn.Linear(C, P_DIM)
        for m in (self.ada_f, self.patch_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t, y):                           # predicts the noise in x
        h = self.patch_in(patchify(x)) + self.pos
        c = self.t_mlp(timestep_embedding(t)) + self.y_emb(y)
        for b in self.blocks:
            h = b(h, c)
        shift, scale = self.ada_f(F.silu(c)).chunk(2, -1)
        return unpatchify(self.patch_out(modulate(self.ln_f(h), shift, scale)))

@torch.no_grad()
def sample(model, y, steps=50, guidance=3.0):
    """DDIM: a deterministic walk from noise to image in `steps` jumps."""
    dev = y.device
    ab = alpha_bar.to(dev)
    x = torch.randn(len(y), 3, 32, 32, device=dev)
    null = torch.full_like(y, len(CLASSES))
    ts = torch.linspace(NSTEPS - 1, 0, steps, device=dev).long()
    for i, t in enumerate(ts):
        tt = t.repeat(len(y))
        eps_c, eps_u = model(x, tt, y), model(x, tt, null)
        eps = eps_u + guidance * (eps_c - eps_u)          # classifier-free guidance
        x0 = (x - (1 - ab[t]).sqrt() * eps) / ab[t].sqrt()            # current guess of the clean image
        x0 = x0.clamp(-1, 1)
        ab_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps          # jump to the next noise level
    return x

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    args = p.parse_args()
    device = pick_device()
    torch.manual_seed(0)
    x_all, y_all = load_cifar()
    model = DiT().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    ab = alpha_bar.to(device)

    for step in range(args.steps + 1):
        i = torch.randint(0, len(x_all), (128,))
        x0, y = x_all[i].to(device), y_all[i].to(device)
        x0 = torch.where(torch.rand(len(x0), 1, 1, 1, device=device) < 0.5, x0, x0.flip(3))  # random mirror
        y = torch.where(torch.rand(len(y), device=device) < 0.1, torch.full_like(y, len(CLASSES)), y)  # 10% unconditional
        t = torch.randint(0, NSTEPS, (len(x0),), device=device)
        noise = torch.randn_like(x0)
        a = ab[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise       # jump straight to noise level t
        loss = F.mse_loss(model(xt, t, y), noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f}")
        if step % 5000 == 0 and step > 0:
            torch.save(model.state_dict(), "dit.pt")

    torch.save(model.state_dict(), "dit.pt")
    model.eval()
    y = torch.arange(len(CLASSES), device=device).repeat_interleave(8)
    grid(to_images(sample(model, y)), ncol=8).save("dit_samples.png")
    print("saved dit.pt and dit_samples.png (one row per class)")
