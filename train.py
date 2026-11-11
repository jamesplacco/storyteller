"""Pretrain the Storyteller GPT. Chapters 7 (loop), 9 (precision) and 10 (distributed).

  python train.py                                   # one device: MPS on a Mac, else CPU
  python train.py --dtype bf16                      # mixed precision (Chapter 9)
  torchrun --nproc_per_node=2 train.py --device cpu # data parallel (Chapter 10)
"""
import argparse, contextlib, math, os, time
import numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import GPT, GPTConfig
from bpe import BPETokenizer

p = argparse.ArgumentParser()
p.add_argument("--device", default="auto")
p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
p.add_argument("--batch_size", type=int, default=32)      # per process, per micro-step
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--grad_accum", type=int, default=1)       # micro-steps per optimizer step
p.add_argument("--max_steps", type=int, default=5000)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--max_lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--n_layer", type=int, default=6)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--out", default="ckpt.pt")
args = p.parse_args()

# ---- distributed setup (Chapter 10). torchrun sets RANK, WORLD_SIZE, LOCAL_RANK ----
ddp = "RANK" in os.environ
if ddp:
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
else:
    rank, world, local = 0, 1, 0
master = rank == 0

if args.device == "auto":
    if torch.cuda.is_available():
        device = f"cuda:{local}"
    elif torch.backends.mps.is_available() and not ddp:
        device = "mps"
    else:
        device = "cpu"
else:
    device = args.device
dev_type = device.split(":")[0]
if dev_type == "cuda":
    torch.cuda.set_device(device)

torch.manual_seed(1337)               # identical init on every rank
np.random.seed(1337 + rank)           # different batches on every rank

# ---- precision (Chapter 9) ----
amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
autocast = torch.autocast(device_type=dev_type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
scaler = torch.amp.GradScaler(dev_type) if args.dtype == "fp16" else None    # fp16 needs loss scaling

B, T = args.batch_size, args.block_size
train = np.memmap("train.bin", dtype=np.uint16, mode="r")
val = np.memmap("val.bin", dtype=np.uint16, mode="r")

def get_batch(split):
    d = train if split == "train" else val
    ix = np.random.randint(0, len(d) - T - 1, B)
    x = torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + T].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def lr_at(step):
    min_lr = args.max_lr / 10
    if step < args.warmup:
        return args.max_lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return min_lr + 0.5 * (args.max_lr - min_lr) * (1 + math.cos(math.pi * p))

vocab_size = len(BPETokenizer().load("tok.pkl").vocab)      # 4,096 with the Chapter 6 tokenizer
cfg = GPTConfig(vocab_size=vocab_size, block_size=T, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
raw = GPT(cfg).to(device)
model = DDP(raw) if ddp else raw
decay = [p for p in raw.parameters() if p.dim() >= 2]
no_decay = [p for p in raw.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8)

@torch.no_grad()
def val_loss(iters=20):
    raw.eval()
    tot = 0.0
    for _ in range(iters):
        with autocast:
            tot += raw(*get_batch("val"))[1].item()
    raw.train()
    return tot / iters

if master:
    n = sum(p.numel() for p in raw.parameters())
    print(f"{n / 1e6:.2f}M params | {device} | {args.dtype} | {world} process(es) | "
          f"{B * T * args.grad_accum * world:,} tokens per step")

t_start, last_val = time.time(), None
for step in range(args.max_steps):
    t0 = time.time()
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for micro in range(args.grad_accum):
        x, y = get_batch("train")
        sync = (not ddp) or micro == args.grad_accum - 1      # all-reduce only on the last micro-step
        with (contextlib.nullcontext() if sync else model.no_sync()):
            with autocast:
                _, loss = model(x, y)
            loss = loss / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
        loss_sum += loss.detach()
    if ddp:
        dist.all_reduce(loss_sum)
        loss_sum /= world
    if scaler:
        scaler.unscale_(opt)                 # back to true gradient scale before clipping
    norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    if scaler:
        scaler.step(opt)                     # skips the step if any gradient is inf or nan
        scaler.update()
    else:
        opt.step()
    if master and step % 50 == 0:
        lv = loss_sum.item()                 # forces the device to finish before timing
        dt = time.time() - t0
        print(f"step {step:5d} | loss {lv:.3f} | norm {norm:.2f} | lr {lr_at(step):.1e} | "
              f"{B * T * args.grad_accum * world / dt:,.0f} tok/s")
    if master and (step % args.eval_every == 0 or step == args.max_steps - 1):
        last_val = val_loss()
        print(f"           val {last_val:.3f}")
        run = {**vars(args), "script": "train.py", "steps_done": step + 1, "world_size": world, "device": device,
               "tokens_seen": (step + 1) * B * T * args.grad_accum * world,
               "seconds": round(time.time() - t_start, 1), "val_loss": round(last_val, 4)}
        torch.save({"model": raw.state_dict(), "cfg": vars(cfg), "train": run}, args.out)

if ddp:
    dist.destroy_process_group()
