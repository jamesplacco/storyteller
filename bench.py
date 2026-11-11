import torch, time
from device import synchronize

def bench(dev, n=4096, iters=20):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    for _ in range(3): a @ b              # warm up
    synchronize(dev); t = time.time()     # GPU calls are asynchronous: wait before timing
    for _ in range(iters): a @ b
    synchronize(dev)
    dt = (time.time() - t) / iters
    return 2 * n ** 3 / dt / 1e12         # TFLOPS

print(f"cpu  {bench('cpu'):.2f} TFLOPS")
if torch.backends.mps.is_available():
    print(f"mps  {bench('mps'):.2f} TFLOPS")
if torch.cuda.is_available():
    print(f"cuda {bench('cuda'):.2f} TFLOPS")
