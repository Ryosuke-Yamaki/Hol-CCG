from utils import circular_correlation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


a = torch.ones(5, requires_grad=True)
b = torch.rand(5)
c = torch.zeros(5)
d = torch.zeros(5)
label = torch.tensor([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float)

criteria = nn.BCELoss(reduction='sum')

loss = criteria(torch.stack((a * b, a * c, a * d)), label)
loss.backward()
print(a.grad)

a = torch.ones(5, requires_grad=True)
loss = criteria(a * c, label[1])
loss.backward()
print(a.grad)

a = torch.ones(5, requires_grad=True)
loss = criteria(a * b, label[0])
loss.backward()
print(a.grad)

a = torch.rand((5, 5))
b = torch.rand((5, 5))
c = circular_correlation(a, b)
print(a)
print(b)
print(c)
print(c.norm(dim=1))
