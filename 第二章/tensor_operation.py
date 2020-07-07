import torch
import numpy as np

# tensor的加法
x = torch.ones(5,3)
y = torch.randn(5,3)
print('x:', x)
print('y:', y)
print('x+y:', x+y)

# Tensor的另一种加法的方式有add方式
print('x+y:', torch.add(x, y))

# Tensor还可以使用inplce进行加法运算
print('x+y:', y.add_(x))

# 使用索引来访问Tensor的的第一行数据
z = y[0, :]
z += 1
print('z:', z)

# 通过view()函数改变Tensor的形状
z = y.view(15)
z1 = y.view(-1,5)
print('y:', y, y.size())
print('z:', z, z.size())
print('z1:', z1, z1.size())

# 使用广播(broadcasting)机制进行计算
x = torch.arange(1,4).view(1,3)
y = torch.arange(1,5).view(4,1)
print('x:', x)
print('y:', y)
print('x+y:', x+y)

# 使用numpy()函数将数组由Tensor转Numpy
x = torch.ones(5,3)
y = x.numpy()
print('x:', x)
print('y:', y)

# 使用from_numpy()函数将数组由Numpy转Tensor
x = np.ones([5,3])
y = torch.from_numpy(x)
print('x:', x)
print('y:', y)

# 使用to()函数可以让Tensor可以在不同的CPU或GPU设备上使用
x = torch.randn([3,5])
print('x', x)
if torch.cuda.is_available():
    device = torch.device('cuda')     #CUDA设备设置
    y = torch.ones_like(x,device=device) #直接在GPU上创建Tensor
    x = x.to(device)  #使用to()函数移动Tensor
    z = x + y
print('x_gpu', x)
print('y_gpu', y)
print('z_gpu', z)
print('z_cpu', z.to('cpu',torch.double))  #to()函数也可以改变数据类型
