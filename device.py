"""Pick the compute device the same way in every script.

  python sft.py                            # auto: NVIDIA GPU, then Apple GPU, then CPU
  STORYTELLER_DEVICE=cpu python sft.py     # force one: auto, cpu, mps or cuda

train.py has its own --device flag because distributed runs give each process its own GPU.
"""
import os, sys, torch

def pick_device(requested=None, verbose=True):
    req = (requested or os.environ.get("STORYTELLER_DEVICE", "auto")).lower()
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if req == "auto":
        dev = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    elif req not in available:
        raise SystemExit(f"STORYTELLER_DEVICE must be auto, cpu, mps or cuda, not {req!r}.")
    elif not available[req]:
        raise SystemExit(f"STORYTELLER_DEVICE={req} was requested, but this machine has no {req} device. "
                         f"Use auto, or one of: {', '.join(d for d, ok in available.items() if ok)}.")
    else:
        dev = req
    if verbose:
        print(f"device: {dev}", file=sys.stderr, flush=True)     # stderr, so piped story text stays clean
    return dev

def synchronize(device):
    """Wait for queued GPU work to finish, so a timer measures the work and not the launch."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
