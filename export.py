"""Export a checkpoint and the tokenizer to flat binary files for run.c (Chapter 8).

  python export.py                 # ckpt.pt -> model.bin, tok.pkl -> tok.bin
  python export.py --ckpt chat.pt  # the chat model works too
"""
import argparse, struct, torch
from bpe import BPETokenizer

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--out", default="model.bin")
args = p.parse_args()

ck = torch.load(args.ckpt, map_location="cpu")
cfg, sd = ck["cfg"], ck["model"]
L = cfg["n_layer"]

with open(args.out, "wb") as f:
    f.write(struct.pack("8i", 0x53544c52, 1, L, cfg["n_head"], cfg["n_embd"],
                        cfg["block_size"], cfg["vocab_size"], 0))   # magic "RLTS", version
    def w(t):
        f.write(t.detach().float().contiguous().numpy().tobytes())
    w(sd["wte.weight"]); w(sd["wpe.weight"])
    # grouped by tensor type across layers, so run.c can index layer l with one offset
    for name in ["ln_1.weight", "ln_1.bias", "attn.qkv.weight", "attn.qkv.bias",
                 "attn.proj.weight", "attn.proj.bias", "ln_2.weight", "ln_2.bias",
                 "mlp.fc.weight", "mlp.fc.bias", "mlp.proj.weight", "mlp.proj.bias"]:
        for l in range(L):
            w(sd[f"blocks.{l}.{name}"])
    w(sd["ln_f.weight"]); w(sd["ln_f.bias"])          # lm_head is tied to wte
print(f"wrote {args.out}")

tok = BPETokenizer().load("tok.pkl")
n = 256 + len(tok.merges) + len(tok.special)
with open("tok.bin", "wb") as f:
    sp = tok.special
    f.write(struct.pack("5i", n, len(tok.merges), sp["<|endoftext|>"], sp["<|user|>"], sp["<|assistant|>"]))
    for i in range(n):
        b = tok.vocab[i]
        f.write(struct.pack("i", len(b))); f.write(b)
    for (a, b), idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        f.write(struct.pack("2i", a, b))
print(f"wrote tok.bin ({n} tokens, {len(tok.merges)} merges)")
