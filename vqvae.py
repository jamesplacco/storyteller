"""An image tokenizer: a VQ-VAE trained on CIFAR-10 (Chapter 17).

A 32x32 color picture becomes an 8x8 grid of integers drawn from a codebook of 512 entries,
so pictures become token sequences, just like text.

  pip install pillow
  python vqvae.py         # -> vqvae.pt, image_codes.pt, recon.png
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
K, D = 512, 64                   # codebook size, code vector size

class VectorQuantizer(nn.Module):
    def __init__(self, K=K, D=D, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1 / K, 1 / K)
        self.beta = beta
        self.register_buffer("usage", torch.zeros(K))           # running count, to find dead codes

    def forward(self, z):                                       # z: (B, N, D) encoder outputs
        W = self.codebook.weight
        d = z.pow(2).sum(-1, keepdim=True) - 2 * z @ W.T + W.pow(2).sum(-1)
        idx = d.argmin(-1)                                      # (B, N) the image tokens
        zq = self.codebook(idx)
        loss = F.mse_loss(zq, z.detach()) + self.beta * F.mse_loss(z, zq.detach())
        if self.training:
            self.usage.mul_(0.99).add_(torch.bincount(idx.flatten(), minlength=W.size(0)).float(), alpha=0.01)
        zq = z + (zq - z).detach()                              # straight-through estimator
        return zq, idx, loss

    @torch.no_grad()
    def restart_dead_codes(self, z):
        """Codes nobody picks get no gradient and stay dead. Move them onto real encoder outputs."""
        dead = (self.usage < 0.01).nonzero().flatten()
        if len(dead):
            flat = z.reshape(-1, z.size(-1))
            self.codebook.weight.data[dead] = flat[torch.randint(0, len(flat), (len(dead),))]
            self.usage[dead] = 1.0
        return len(dead)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 1))
    def forward(self, x):
        return x + self.net(x)

class VQVAE(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, ch, 4, 2, 1), nn.ReLU(),           # 32 -> 16
                                 nn.Conv2d(ch, ch, 4, 2, 1),                      # 16 -> 8
                                 ResBlock(ch), ResBlock(ch), nn.ReLU(), nn.Conv2d(ch, D, 1))
        self.vq = VectorQuantizer()
        self.dec = nn.Sequential(nn.Conv2d(D, ch, 3, padding=1), ResBlock(ch), ResBlock(ch), nn.ReLU(),
                                 nn.ConvTranspose2d(ch, ch, 4, 2, 1), nn.ReLU(),  # 8 -> 16
                                 nn.ConvTranspose2d(ch, 3, 4, 2, 1))              # 16 -> 32

    def encode(self, x):                                        # (B, 3, 32, 32) -> z (B, 64, D)
        return self.enc(x).flatten(2).transpose(1, 2)

    def decode_z(self, zq):                                     # (B, 64, D) -> (B, 3, 32, 32)
        B, N, _ = zq.shape
        s = int(N ** 0.5)
        return self.dec(zq.transpose(1, 2).reshape(B, D, s, s))

    def decode_codes(self, idx):                                # (B, 64) ints -> images
        return self.decode_z(self.vq.codebook(idx))

    def forward(self, x):
        z = self.encode(x)
        zq, idx, vq_loss = self.vq(z)
        return self.decode_z(zq), idx, vq_loss, z

def load_vqvae(path, device):
    m = VQVAE().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()

def to_images(x):
    """(B, 3, 32, 32) in [-1, 1] -> list of PIL images."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in x]

def grid(images, ncol, scale=3):
    w, h = images[0].size
    nrow = (len(images) + ncol - 1) // ncol
    g = Image.new("RGB", (ncol * w, nrow * h), "white")
    for i, im in enumerate(images):
        g.paste(im, ((i % ncol) * w, (i // ncol) * h))
    return g.resize((g.width * scale, g.height * scale), Image.NEAREST)

def load_cifar(split="train"):
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split=split)
    x = np.stack([np.asarray(im.convert("RGB")) for im in ds["img"]])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 127.5 - 1      # (N, 3, 32, 32) in [-1, 1]
    return x, torch.tensor(ds["label"])

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    x_all, y_all = load_cifar()
    print(f"{len(x_all):,} images")
    model = VQVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    var = x_all.var().item()

    for step in range(6001):
        x = x_all[torch.randint(0, len(x_all), (128,))].to(device)
        recon, idx, vq_loss, z = model(x)
        rec_loss = F.mse_loss(recon, x) / var
        loss = rec_loss + vq_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 and step > 0:
            dead = model.vq.restart_dead_codes(z.detach())
        if step % 500 == 0:
            used = (model.vq.usage > 0.01).sum().item()
            print(f"step {step:5d} | recon {rec_loss.item():.4f} | vq {vq_loss.item():.4f} | codes in use {used}/{K}")

    model.eval()
    torch.save(model.state_dict(), "vqvae.pt")
    codes = []
    with torch.no_grad():                                       # tokenize the whole dataset once
        for i in range(0, len(x_all), 1000):
            codes.append(model.vq(model.encode(x_all[i:i + 1000].to(device)))[1].cpu())
        x = x_all[:16].to(device)
        recon = model(x)[0]
    torch.save({"codes": torch.cat(codes).short(), "labels": y_all}, "image_codes.pt")
    grid(to_images(torch.cat([x, recon])), ncol=16).save("recon.png")
    print("saved vqvae.pt, image_codes.pt, recon.png (top row: originals, bottom: from 64 tokens each)")
