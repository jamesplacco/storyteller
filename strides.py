import torch
x = torch.arange(12).view(3, 4)
print(x.stride())                  # (4, 1): next row is 4 elements away
t = x.T
print(t.stride(), t.is_contiguous())   # (1, 4) False: same memory, new metadata
print(t.contiguous().stride())     # (3, 1): copied into row-major order