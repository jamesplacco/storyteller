import math, torch, torch.nn as nn

def softmax(z):
    z = z - z.max(dim=-1, keepdim=True).values   # subtract max for numerical stability
    e = z.exp()
    return e / e.sum(dim=-1, keepdim=True)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)                  # each (B, T, hs)
        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))      # (B, T, T) scores
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = softmax(att)
        self.last_att = att.detach()
        return att @ v                                              # (B, T, hs)

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 1, 6, 32
    tok_emb = torch.randn(B, T, C)
    pos_emb = nn.Embedding(T, C)
    x = tok_emb + pos_emb(torch.arange(T))     # inject order
    head = Head(C, 16, T)
    out = head(x)
    print(out.shape)
    print(head.last_att[0].round(decimals=2))