import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset

device = "mps" if torch.backends.mps.is_available() else "cpu"
ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
text = "\n".join(ds["text"])
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

CTX, EMB, HID = 8, 24, 256

def get_batch(B=256):
    ix = torch.randint(len(data) - CTX - 1, (B,))
    x = torch.stack([data[i:i + CTX] for i in ix])   # (B, CTX) previous chars
    y = data[ix + CTX]                                # (B,)    next char
    return x.to(device), y.to(device)

def gelu(x):  # what F.gelu(approximate="tanh") computes
    return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))

class NGramMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.fc1 = nn.Linear(CTX * EMB, HID)
        self.fc2 = nn.Linear(HID, V)
    def forward(self, x):
        e = self.emb(x).view(x.size(0), -1)   # (B, CTX*EMB): concatenate embeddings
        return self.fc2(gelu(self.fc1(e)))    # (B, V) logits

model = NGramMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
for step in range(5001):
    x, y = get_batch()
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step:5d}  loss {loss.item():.3f}")

ctx = [stoi[" "]] * CTX
out = []
with torch.no_grad():
    for _ in range(300):
        logits = model(torch.tensor([ctx], device=device))
        ix = torch.multinomial(F.softmax(logits, -1), 1).item()
        out.append(itos[ix]); ctx = ctx[1:] + [ix]
print("".join(out))