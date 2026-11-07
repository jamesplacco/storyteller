# Storyteller

[![tests](https://github.com/jamesplacco/storyteller/actions/workflows/tests.yml/badge.svg)](https://github.com/jamesplacco/storyteller/actions/workflows/tests.yml)

A complete build of Andrej Karpathy's LLM101n syllabus: a small language model that writes and
illustrates bedtime stories, built from scratch across 17 chapters. The course site explains each
file; this folder holds every file in its final, tested form.

Written for an Apple Silicon Mac (PyTorch `mps`), with automatic fallback to CPU. Runs unchanged
on NVIDIA GPUs.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TinyStories (about 2 GB) and CIFAR-10 download from Hugging Face the first time a script needs them.

## Check your setup first

```bash
pip install pytest httpx
pytest -x -v tests
```

This runs every chapter's script, in order, at toy scale on synthetic stand-in data: no downloads,
3 to 10 minutes. Each test checks a claim the course makes, for example that the C engine prints
exactly what PyTorch prints, that the KV-cache changes nothing but speed, and that hand-written
ZeRO-1 matches AdamW. On a Mac it also checks that the Apple GPU gives the same answers as the CPU.
The same suite runs on GitHub on every push.

## Run order

Each step's output feeds the next. Times are estimates for an M-series Mac.

| Ch | Command | Makes | Time |
|----|---------|-------|------|
| 01 | `python bigram.py` | | seconds |
| 02 | `python micrograd.py` | | seconds |
| 03 | `python mlp.py` | | minutes |
| 04 | `python attention.py` | | seconds |
| 05 | `python -c "from model import GPT, GPTConfig; print(sum(p.numel() for p in GPT(GPTConfig()).parameters()))"` | | seconds |
| 06 | `python train_tok.py` then `python prepare.py` | tok.pkl, train.bin, val.bin | 15 to 45 min |
| 07 | `python train.py` then `python sample.py` | ckpt.pt | 30 to 90 min |
| 08 | `python strides.py`, `python bench.py`, `clang -O3 -ffast-math -o matmul matmul.c && ./matmul` | | seconds |
| 08 | `python export.py`, `clang -O3 -ffast-math -o run run.c -lm`, `./run model.bin tok.bin` | model.bin, tok.bin | seconds |
| 09 | `python precision.py`, optionally `python train.py --dtype bf16` | | 1 min |
| 10 | `torchrun --nproc_per_node=2 ddp_demo.py` (also `zero1.py`, `fsdp_demo.py`) | | 1 min each |
| 11 | `python make_sft.py` | sft.json | minutes |
| 12 | `python bench_kv.py` | | seconds |
| 13 | `python quantize.py` | | seconds |
| 14 | `python sft.py` | chat.pt | 10 to 30 min |
| 15 | `python reward.py`, `python ppo.py`, `python dpo.py` | reward.pt, chat_ppo.pt, chat_dpo.pt | 1 to 2 h total |
| 15 | `python eval.py` | a table comparing every checkpoint | 5 to 10 min |
| 17 | `python vqvae.py`, `python image_gpt.py`, `python dit.py` | vqvae.pt, image_gpt.pt, dit.pt | 1 to 3 h total |
| 16 | `uvicorn app:app --port 8000`, then open http://localhost:8000 | | |

The app uses the best chat checkpoint it finds (`chat_ppo.pt`, then `chat_dpo.pt`, then `chat.pt`);
set `CKPT=...` to choose. It draws illustrations once `vqvae.pt` and `image_gpt.pt` exist.

## Notes

- Long scripts take `--steps` (`--iters` for `ppo.py`) if you want a shorter run first.
- `model.py` already includes Chapter 12's KV-cache. `train.py` is the final version from
  Chapters 7, 9 and 10: `--dtype bf16|fp16`, `--grad_accum N`, and DDP under `torchrun`.
- On a Mac, the distributed scripts run as CPU processes with the `gloo` backend. On NVIDIA
  GPUs they switch to `nccl` and `cuda` automatically.
- If bf16 autocast isn't supported on your macOS and PyTorch versions, train in fp32.
- `./run model.bin tok.bin -t 0` and `python sample.py --temperature 0` print the same story.
