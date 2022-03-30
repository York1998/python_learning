# 作者：York
# 时间：2022/3/28 22:27
from __future__ import  print_function
import torch

x = torch.empty(5, 3)
print(x)

X = torch.zeros(5, 3, dtype=torch.long)
print(X)

y = torch.tensor([5.5, 3])
print(y)

z = x.new_ones(5, 3, dtype=torch.double)
print(z)

x = torch.randn_like(x,dtype=torch.float)
print(x)

#获取它的形状
print(x.size())

#可以使用像标准的Numpy一样的各种索引操作
print(x[:,1])
print(x[1,:])

#改变形状
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
w = x.view(-1)
print(x.size(), y.size(), z.size())
print(z,z.size())

#使用.item()来获得对应的Python数值
#print(x.item()) 如果是仅包含一个元素的tensor，可以使用.item()来得到对应的python数值

" torch convert to numpy"
a = torch.ones(5)
print("a is {}".format(a))
b = a.numpy()
print(b)

" numpy convert to torch "
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)

print(b)

"张量可以使用.to方法移动到任何设备（device）上："

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))