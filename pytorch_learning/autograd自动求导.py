# 作者：York
# 时间：2022/3/29 11:12
import torch
"""
    pytorch中，所有神经网络的核心是autograd宝
    autograd包为张量上的所有操作提供了自动求导机制。是一个在运行时定义（define-by-run）的框架，这意味着反向传播
    是根据代码如何运行来决定的。并且每次迭代可以是不同的。
    
    torch.Tensor 是这个包的核心类，必须要设置它的属性为 .requires_grad 为 True，那么将会追踪对于该向量的所有
    操作。当完成计算后可以调用 .backward() 来自动计算所有的梯度。所有梯度将会自动类驾到.grad属性上去。
    
    要组织一个张量被跟踪历史，可以调用.detach()方法将其与历史分离，并组织未来的计算记录被跟踪。
    为了防止跟踪历史记录（和使用内存），可以将代码包装在with torch.no_grad(): 中。
    在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练的参数，
    但是我们不需要在此过程中对他们进行梯度计算。
"""

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn) #y是计算的结果，所以它有grad_fn属性。

z = y * y * 3
out = z.mean()
print(z, out)

# out.backward(torch.tensor(2.)) #这样计算的是全为9
out.backward(torch.tensor(1.)) #这样计算的是全为4.5

print(x.grad) # 得4.5 因为，out.backward()和out.backward(torch.tensor(1.))等价

"""
    通常来说，**torch.autograd **是计算雅可比向量积的一个“**引擎**”。根据链式法则，雅可比向量积：
    雅可比向量积的这一特性使得将**外部梯度输入到具有非标量输出的模型中**变得非常方便。

    现在我们来看一个雅可比向量积的例子:
"""
"""
    torch.rand(*sizes, out=None) → Tensor
        返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。
        张量的形状由参数sizes定义。
    torch.randn(*sizes, out=None) → Tensor
        返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。
        张量的形状由参数sizes定义。
"""
x = torch.randn(3, requires_grad=True)
y = x * 2
"""
    data.norm() L2范数
"""
print(y)
print(y.data)
while y.data.norm() < 1000:
    y = y * 2
    print(y)
print(y)


