"""ZeRO stage 3 with PyTorch FSDP (Chapter 10). Every weight is split across ranks and
gathered just in time for each block's forward and backward.

  torchrun --nproc_per_node=2 fsdp_demo.py
"""
import torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from model import GPT, GPTConfig

dist.init_process_group("gloo")                  # "nccl" on NVIDIA GPUs
rank = dist.get_rank()
torch.manual_seed(0)
model = GPT(GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=4, n_embd=128))
for block in model.blocks:
    fully_shard(block)                           # each block gathers its own weights when needed
fully_shard(model)                               # embeddings and final norm
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # optimizer state is sharded too

g = torch.Generator().manual_seed(rank)
x = torch.randint(0, 512, (4, 65), generator=g)  # a fixed batch per rank, so loss should fall
for step in range(30):
    _, loss = model(x[:, :-1], x[:, 1:])
    opt.zero_grad(); loss.backward(); opt.step()
    if rank == 0 and step % 10 == 0:
        print(f"step {step:2d}  loss {loss.item():.3f}")

w = model.wte.weight
print(f"rank {rank}: wte is {tuple(w.shape)} logically, this rank stores {tuple(w.to_local().shape)}")
dist.destroy_process_group()
