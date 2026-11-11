"""ZeRO stage 1 by hand (Chapter 10). Each rank keeps AdamW state for only its slice of the
parameters, updates that slice, then all-gathers the slices so every rank has the full model.
Rank 0 also trains an ordinary AdamW copy on the whole batch and checks the two agree.

  torchrun --nproc_per_node=2 zero1.py
"""
import torch, torch.distributed as dist
from model import GPT, GPTConfig

dist.init_process_group("gloo")                  # "nccl" on NVIDIA GPUs
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)
cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=4, n_embd=128)
model = GPT(cfg)                                 # same seed, so the same init on every rank
params = list(model.parameters())                # tied wte/lm_head appears once
n = sum(p.numel() for p in params)
shard = (n + world - 1) // world                 # pad so every rank owns an equal slice
lo, hi = rank * shard, min(n, (rank + 1) * shard)

flat = torch.nn.utils.parameters_to_vector(params).detach()
master = flat[lo:hi].clone()                     # this rank's slice of the weights
m, v = torch.zeros_like(master), torch.zeros_like(master)   # Adam state: only for this slice
lr, b1, b2, eps, wd = 1e-3, 0.9, 0.95, 1e-8, 0.1

if rank == 0:                                    # reference: plain AdamW, full batch, one process
    ref = GPT(cfg); ref.load_state_dict(model.state_dict())
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

def global_batch(step, B=8, T=64):
    g = torch.Generator().manual_seed(step)
    x = torch.randint(0, cfg.vocab_size, (B, T + 1), generator=g)
    return x[:, :-1], x[:, 1:]

for step in range(1, 21):
    x, y = global_batch(step % 4)                            # 4 batches on repeat, so loss falls
    _, loss = model(x[rank::world], y[rank::world])          # each rank: its share of the batch
    model.zero_grad()
    loss.backward()
    g = torch.cat([p.grad.view(-1) for p in params])
    dist.all_reduce(g)                                       # NCCL would use reduce_scatter and send
    g = g[lo:hi] / world                                     # each rank only the slice it owns

    master.mul_(1 - lr * wd)                                 # AdamW on this slice only
    m.mul_(b1).add_(g, alpha=1 - b1)
    v.mul_(b2).addcmul_(g, g, value=1 - b2)
    master.addcdiv_(m / (1 - b1 ** step), (v / (1 - b2 ** step)).sqrt() + eps, value=-lr)

    piece = torch.zeros(shard); piece[:hi - lo] = master
    parts = [torch.zeros(shard) for _ in range(world)]
    dist.all_gather(parts, piece)                            # everyone gets everyone's slice
    torch.nn.utils.vector_to_parameters(torch.cat(parts)[:n], params)

    if rank == 0:
        _, ref_loss = ref(x, y)
        ref_opt.zero_grad(); ref_loss.backward(); ref_opt.step()
        if step % 5 == 0:
            print(f"step {step:2d}  loss {ref_loss.item():.4f}")

if rank == 0:
    diff = (torch.nn.utils.parameters_to_vector(params) - torch.nn.utils.parameters_to_vector(ref.parameters())).abs().max()
    print(f"max |ZeRO-1 weights - AdamW weights| = {diff.item():.1e}")
    print(f"Adam state per rank: {2 * shard * 4 / 1e6:.2f} MB with ZeRO-1 vs {2 * n * 4 / 1e6:.2f} MB without")
dist.destroy_process_group()
