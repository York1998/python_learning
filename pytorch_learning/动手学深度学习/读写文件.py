# 作者：York
# 时间：2022/4/4 13:25
"""
    有时候我们需要保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）
    此外，当运行一个耗时很长的训练模型时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，
    不会损失几天的计算结果。因此，学习如何加载和存储向量和整个模型是很重要的。
"""

"""
    1、加载和保存张量
"""
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)  #结果不包括end
torch.save(x, 'x-file')
print(f"存入的模型的张量为{x}")

#将存储的一个张量列表 来写回内存
x2 = torch.load('x-file')
print(f"写回的模型的张量为{x2}")

#我们可以存储一个张量列表，然后把它们读回内存。
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
print(x2, y2)


#甚至可以写入或读取从字符串映射到张量的字典。当我们尧都区或写入模型中的所有权重时，很方便
mydict = {'x':x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

"""
    2、加载和保存模型参数
        保存单个权重向量确实有用，但是我们如果想保存整个模型，并在以后加载它们，单独保存每个向量
        则会变得很麻烦。毕竟有很多参数遍布在各处。
        因此，深度学习框架提供了内置函数来保存和加载整个网络。需要注意的一个细节是：这将保存模型的
        参数而不是保存整个模型。
        
        例如：如果有一个3层感知机，需要单独制定框架。因为模型本身可以包含任意代码，所以模型本身难以序列化。
            因此为了回复模型，需要用代码生成框架，然后从磁盘加载数据，以下从熟悉的多层感知机开始尝试一下。
"""

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self,x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)

#接下来，我们把模型的参数存储在一个叫做"mpl.params"的文件中。
torch.save(net.state_dict(), 'mlp.params')

#为了回复模型，实例化原始多层感知机模型的一个备份。这里不需要随机初始化模型参数，而是直接读取文件中存储的参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

#由于两个实例具有相同的模型参数，在输入相同的X时， 两个实例的计算结果应该相同。 让我们来验证一下。

Y_clone = clone(X)
print(Y_clone == Y)

