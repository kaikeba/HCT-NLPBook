import torch
print(torch.__version__)

# 通过torch来构造一个5*3的矩阵
x = torch.empty(5,3)
print(x)

# 构造一个随机初始化的5*3矩阵
x = torch.rand(5,3)
print(x)

# 构造一个全都是0的5*3的矩阵
x = torch.zeros(5,3,dtype=torch.long)
print(x)

# 使用已有的数据来构造一个新的Tensor
x = torch.tensor([5,3])
print(x)

# 构造基于已经存在的Tensor下新的Tensor
x = x.new_ones(5,3)
print(x)
x1 = torch.randn_like(x,dtype=torch.float64)
print(x1)
