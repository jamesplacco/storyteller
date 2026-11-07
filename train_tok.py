from datasets import load_dataset
from bpe import BPETokenizer

ds = load_dataset("roneneldan/TinyStories", split="train[:100000]")
tok = BPETokenizer()
tok.train("\n".join(ds["text"]), vocab_size=4096, verbose=True)
tok.save("tok.pkl")

s = "Once upon a time, there was a little dog named Max."
ids = tok.encode(s)
print(f"{len(s)} chars -> {len(ids)} tokens: {[tok.decode([i]) for i in ids]}")
assert tok.decode(ids) == s