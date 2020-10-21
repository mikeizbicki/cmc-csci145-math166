#!/usr/bin/python3

import torch

def f(x):
    return x**2 + 4*x + 2

x = torch.tensor(-2.0)
x.requires_grad=True
z = f(x)
z.backward()
print("x.grad=",x.grad)
