import torch, time

def bench(dev, n=4096, iters=20):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    sync = torch.mps.synchronize if dev == "mps" else (lambda: None)
    for _ in range(3): a @ b              # warm up
    sync(); t = time.time()
    for _ in range(iters): a @ b
    sync()
    dt = (time.time() - t) / iters
    return 2 * n ** 3 / dt / 1e12         # TFLOPS

print(f"cpu {bench('cpu'):.2f} TFLOPS")
if torch.backends.mps.is_available():
    print(f"mps {bench('mps'):.2f} TFLOPS")