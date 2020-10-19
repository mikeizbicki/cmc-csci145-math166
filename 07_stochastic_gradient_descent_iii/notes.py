#!/usr/bin/python3

print('test')

import torch

x = torch.Tensor([1,2,3])

print("x=",x)
print("x.shape=",x.shape)

M = torch.Tensor([[1,2,3],[3,4,5]])
print("M.shape=",M.shape)

print("M @ x=",M @ x)
#print("x @ M=",x @ M)


# pytorch provides automatic differentiation
# different than symbolic differentiation and numeric differentiation

def f(x):
    r = 1
    for i in range(2):
        r*=x
    return r + 4*x + 2
    #return x**2 + 4*x + 2

'''
def df(x):
    return 2*x + 4
'''

print("f(4)=",f(4))

a = torch.tensor(4)
print("f(a)=",f(a))

# when x = -2, then the function is minimized

x = torch.tensor(-2.0)
x.requires_grad=True
z = f(x)
z.backward()
print("x.grad=",x.grad)
#print("df(x)=",df(x))

print("f(-2)=",f(-2))
print("f(-2.1)=",f(-2.1))
print("f(-1.9)=",f(-1.9))

eta = 0.1
x_t = torch.tensor(0.0)
x_t.requires_grad = True

result = f(x_t)
result.backward()
print("x_t.grad=",x_t.grad)
#print("df(x_t)=",df(x_t))

print("x_t=",x_t)
for i in range(100):
    # x_t = x_t - eta * df(x_t)
    x_t.requires_grad = True
    result = f(x_t)
    result.backward()   # computes all the gradients
    with torch.no_grad():
        x_t = x_t - eta * x_t.grad    # df(x_t) = x_t.grad
    print("x_t=",x_t)
