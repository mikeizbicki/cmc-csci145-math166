import torch

# install with pip3 install torch

v = torch.Tensor([1,2,3,4])

m = torch.Tensor([[1,2],[3,4]])

v.shape
m.shape

# componentwise multiplication
v * v

# @ symbol for matrix multiplication


y = torch.ones([3,4])
x = torch.zeros([1,4])

x.t() # transpose

# this all only works on dense matrices

# how do you construct a sparse matrix?

i = torch.LongTensor([[0,1,1],[2,0,2]])
v = torch.FloatTensor([3,4,5])
z = torch.sparse.FloatTensor(i,v, torch.Size([2,3]))

torch.sparse.mm # function for matrix - matrix multiplication, first matrix is sparse


# equivalent
torch.sparse.mm(z.t(), m.t()).t()
torch.sparse.addmm
m @ z.to_dense()
