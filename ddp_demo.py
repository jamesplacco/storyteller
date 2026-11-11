import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig

dist.init_process_group("gloo")                  # CPU backend; use "nccl" on NVIDIA
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)
model = DDP(GPT(GPTConfig(n_layer=2, n_head=4, n_embd=128)))   # broadcasts rank 0's weights
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(20):
    g = torch.Generator().manual_seed(step * world + rank)       # different data per rank
    seq = torch.randint(0, 4096, (8, 65), generator=g)
    x, y = seq[:, :-1], seq[:, 1:]                               # next-token targets
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()                # backward all-reduces grads
    if rank == 0 and step % 5 == 0:
        print(f"step {step} loss {loss.item():.3f}")

flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach()   # every weight
mx, mn = flat.clone(), flat.clone()
dist.all_reduce(mx, op=dist.ReduceOp.MAX)
dist.all_reduce(mn, op=dist.ReduceOp.MIN)
if rank == 0:
    print("replicas identical:", torch.equal(mx, mn))
dist.destroy_process_group()