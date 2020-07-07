import torch
from torch.autograd import Variable

# 设置requires_grad = True来跟踪与它相关的计算
x = torch.randn(5, 3, requires_grad = True)
print('x:', x)
print(x.grad_fn)

# 对Tensor x做加法运算
y = x + 1
print('y：', y)
print(y.grad_fn)

# 对Tensor y 做更复杂的运算
z = y * y * 3
out = z.mean()
print('z:', z)
print('out:', out)

# .requires_grad_( ... ) 会改变张量的 requires_grad 标记。输入的标记默认为 False 
x = torch.randn(5, 3)
x = ((x * 3) / (x - 1))
print('x.requires_grad:', x.requires_grad)
x.requires_grad_(True)
print('x.requires_grad:', x.requires_grad)
y = (x * x).sum()
print('y.grad_fn:', y.grad_fn)

# 梯度计算
x = torch.Tensor([[1.,2.,3.], [4.,5.,6.]])
x = Variable(x, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print('x.grad:', x.grad)

# 简单的雅可比的梯度
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print('y:', y)


v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print('x.grad:', x.grad)

print('x.requires_grad:', x.requires_grad)
print('(x ** 2).requires_grad:', (x ** 2).requires_grad)
with torch.no_grad():
    print('(x ** 2).requires_grad:', (x ** 2).requires_grad)
