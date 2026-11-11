"""Read a story from a checkpoint (Chapter 7). Uses the KV-cache from Chapter 12.

  python sample.py
  python sample.py --ckpt chat.pt --chat "Tell me a story about a frog."
"""
import argparse, torch
from model import GPT, GPTConfig
from bpe import BPETokenizer
from device import pick_device

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="ckpt.pt")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--chat", default=None, help="wrap this request in the chat template")
p.add_argument("--n", type=int, default=200)
p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
args = p.parse_args()

device = pick_device()
tok = BPETokenizer().load("tok.pkl")
ck = torch.load(args.ckpt, map_location=device)
model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
model.load_state_dict(ck["model"])

text = f"<|user|>{args.chat}<|assistant|>" if args.chat else args.prompt
idx = torch.tensor([tok.encode(text)], device=device)
greedy = args.temperature == 0                           # always pick the top token, like ./run -t 0
out = model.generate_cached(idx, args.n, temperature=1.0 if greedy else args.temperature, top_k=1 if greedy else 50)
out = tok.decode(out[0].tolist())
if args.chat:
    out = out.split("<|assistant|>", 1)[1]
print(out.split("<|endoftext|>")[0])
