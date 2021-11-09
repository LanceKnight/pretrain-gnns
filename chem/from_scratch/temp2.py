import torch

a = torch.tensor([1, 1, 1])
print(a.shape)
b = torch.tensor([2, 2, 2])
print(b.shape)
c = torch.cat([a, b], dim=0)
print(c)
