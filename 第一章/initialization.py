import torch
import math

def relu(x):
    return x.clamp_min(0.)

mean, var = 0., 0.
for i in range(10000):
    x = torch.randn(512)
    a = torch.randn(512,512)
    y = relu(a @ x)
    mean += y.mean().item()
    var += y.pow(2).mean().item()
print('均值：', mean/10000, '平均标准差：', math.sqrt(var/10000))
print('输入连接层数的平方根：', math.sqrt(512/2))

# 按输入连接层数的平方根缩放权重矩阵W
mean, var = 0., 0.
for i in range(10000):
    x = torch.randn(512)
    w = torch.randn(512,512) * math.sqrt(2/512)
    y = relu(w @ x)
    mean += y.mean().item()
    var += y.pow(2).mean().item()
print('均值：', mean/10000,' 平均标准差：', math.sqrt(var/10000))

def kaiming(m,h):
    return torch.randn(m,h)*math.sqrt(2./m)

# 使用kaiming初始化
x = torch.randn(512)
for i in range(100):
    w = kaiming(512,512)
    x = relu(w @ x)
print(x.mean(),x.std())
