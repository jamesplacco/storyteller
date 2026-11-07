import torch
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
text = "\n".join(ds["text"])

chars = sorted(set(text))                    # the vocabulary
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)
data = torch.tensor([stoi[c] for c in text])

# count every (current, next) pair in one shot
N = torch.zeros((V, V))
N.index_put_((data[:-1], data[1:]), torch.ones(len(data) - 1), accumulate=True)

P = (N + 1) / (N + 1).sum(dim=1, keepdim=True)   # add-one smoothing, rows sum to 1

nll = -P[data[:-1], data[1:]].log().mean()
print(f"vocab {V} | avg NLL {nll.item():.3f} | uniform would be {torch.log(torch.tensor(V)).item():.3f}")

g = torch.Generator().manual_seed(42)
ix, out = stoi["O"], ["O"]
for _ in range(300):
    ix = torch.multinomial(P[ix], 1, generator=g).item()
    out.append(itos[ix])
print("".join(out))