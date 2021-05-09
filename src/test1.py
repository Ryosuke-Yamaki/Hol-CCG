import numpy as np
import torch
import csv
import random
from torch.fft import fft, ifft
from torch import conj, mul


def circular_correlation(a, b):
    a = conj(fft(a))
    b = fft(b)
    c = mul(a, b)
    c = ifft(c).real
    return c.div(c.norm(dim=1, keepdim=True) + 1e-6)


a = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [2, 3, 4, 5, 6]])
b = torch.tensor([[5, 4, 3, 2, 1], [1, 2, 3, 4, 5], [6, 5, 4, 3, 2]])

c = circular_correlation(a, b)
print(c)
