"""Train a reward model on preference pairs (Chapter 15). The first half of classic RLHF.

The reward model is the chat GPT with its output layer swapped for a single number. Trained
with the Bradley-Terry loss: P(chosen beats rejected) = sigmoid(r_chosen - r_rejected).

  python reward.py        # needs chat.pt and sft.json -> reward.pt
"""
import argparse, json, random, torch, torch.nn as nn, torch.nn.functional as F
from model import GPT, GPTConfig
from bpe import BPETokenizer

T = 256

class RewardModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gpt = GPT(GPTConfig(**cfg))
        self.head = nn.Linear(cfg["n_embd"], 1)

    def forward(self, idx, last):
        """idx: (B, T) token ids. last: (B,) position of each sequence's final real token."""
        g = self.gpt
        x = g.wte(idx) + g.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in g.blocks:
            x, _ = block(x)
        h = g.ln_f(x)[torch.arange(idx.size(0), device=idx.device), last]
        return self.head(h).squeeze(-1)          # one score per sequence

def encode(tok, prompt, response):
    ids = tok.encode(f"<|user|>{prompt}<|assistant|>") + tok.encode_ordinary(response) + [tok.special["<|endoftext|>"]]
    return ids[:T]

def pad(seqs, fill):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [fill] * (L - len(s)) for s in seqs]), torch.tensor([len(s) - 1 for s in seqs])

def load_reward_model(path, device):
    ck = torch.load(path, map_location=device)
    rm = RewardModel(ck["cfg"]).to(device)
    rm.load_state_dict(ck["model"])
    return rm.eval()

def make_pair(data):
    ex = random.choice(data)
    for _ in range(100):                                   # find a story about something else
        bad = random.choice(data)
        if ex["word"] not in bad["response"].lower():
            break
    return ex["prompt"], ex["response"], bad["response"]   # prompt, chosen, rejected

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    args = p.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    random.seed(0); torch.manual_seed(0)
    tok = BPETokenizer().load("tok.pkl")
    EOT = tok.special["<|endoftext|>"]

    ck = torch.load("chat.pt", map_location=device)
    T = min(T, ck["cfg"]["block_size"])                 # never longer than the model's context
    rm = RewardModel(ck["cfg"]).to(device)
    rm.gpt.load_state_dict(ck["model"])          # start from the chat model's understanding of text

    data = json.load(open("sft.json"))
    split = int(0.9 * len(data))
    train_data, test_data = data[:split], data[split:]

    def batch(src, n=8):
        pairs = [make_pair(src) for _ in range(n)]
        seqs = [encode(tok, p, r) for p, c, rj in pairs for r in (c, rj)]   # chosen, rejected, ...
        idx, last = pad(seqs, EOT)
        return idx.to(device), last.to(device)

    opt = torch.optim.AdamW(rm.parameters(), lr=5e-5, weight_decay=0.0)
    for step in range(args.steps):
        idx, last = batch(train_data)
        r = rm(idx, last)
        loss = -F.logsigmoid(r[0::2] - r[1::2]).mean()       # Bradley-Terry
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}  train acc {(r[0::2] > r[1::2]).float().mean().item():.2f}")

    rm.eval(); correct = total = 0
    with torch.no_grad():
        for _ in range(25):
            idx, last = batch(test_data)
            r = rm(idx, last)
            correct += (r[0::2] > r[1::2]).sum().item()
            total += r.size(0) // 2
    print(f"held-out accuracy {correct / total:.2f} on {total} pairs of stories it never trained on")
    torch.save({"model": rm.state_dict(), "cfg": ck["cfg"]}, "reward.pt")
