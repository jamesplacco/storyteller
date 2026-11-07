"""Image GPT: the Chapter 5 transformer, unchanged, trained on image tokens (Chapter 17).

Each training sequence is [class token] + 64 image tokens from vqvae.py. Vocabulary is the 512
codebook entries plus 10 class tokens. Generating after a class token draws a new picture.

  python image_gpt.py     # needs image_codes.pt, vqvae.pt -> image_gpt.pt, samples.png
"""
import argparse, math, torch
from model import GPT, GPTConfig
from vqvae import K, CLASSES, load_vqvae, to_images, grid

def generate_codes(model, cls, n, temperature=1.0, top_k=100):
    idx = torch.full((n, 1), K + cls, dtype=torch.long, device=next(model.parameters()).device)
    out = model.generate_cached(idx, 64, temperature=temperature, top_k=top_k)[:, 1:]
    return out.clamp(max=K - 1)            # class tokens are never targets, but be safe

def load_image_gpt(path, device):
    ck = torch.load(path, map_location=device)
    m = GPT(GPTConfig(**ck["cfg"])).to(device)
    m.load_state_dict(ck["model"])
    return m.eval()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000)
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    data = torch.load("image_codes.pt")
    seqs = torch.cat([data["labels"].long().unsqueeze(1) + K, data["codes"].long()], dim=1)   # (N, 65)
    cfg = GPTConfig(vocab_size=K + len(CLASSES), block_size=65, n_layer=6, n_head=6, n_embd=384)
    model = GPT(cfg).to(device)
    steps, warmup, max_lr = args.steps, 200, 6e-4
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    for step in range(steps):
        lr = max_lr * min(1, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr
        b = seqs[torch.randint(0, len(seqs), (64,))].to(device)
        _, loss = model(b[:, :-1], b[:, 1:])          # predict every image token from what precedes it
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"step {step:5d} | loss {loss.item():.3f} | lr {lr:.1e}")
    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, "image_gpt.pt")

    model.eval()
    vq = load_vqvae("vqvae.pt", device)
    with torch.no_grad():
        imgs = [im for c in range(len(CLASSES)) for im in to_images(vq.decode_codes(generate_codes(model, c, 8)))]
    grid(imgs, ncol=8).save("samples.png")
    print("saved image_gpt.pt and samples.png (one row per class: " + ", ".join(CLASSES) + ")")
