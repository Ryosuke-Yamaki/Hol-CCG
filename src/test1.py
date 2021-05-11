import numpy as np
import torch
import csv
import random
from torch.fft import fft, ifft
from torch import conj, mul
import time

start = time.time()
m = []
for i in range(10000000):
    m.append(i)
print(time.time() - start)

start = time.time()
m = [i for i in range(10000000)]
print(time.time() - start)
