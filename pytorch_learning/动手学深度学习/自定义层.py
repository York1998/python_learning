# 作者：York
# 时间：2022/4/4 12:53
import torch
import torch.nn.functional as F
from torch import nn
"""
    深度学习成功背后的一个因素是神经网络的灵活性： 我们可以用创造性的方式组合不同的层，
    从而设计出适用于各种任务的架构。
    例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。 
    未来，你会遇到或要自己发明一个现在在深度学习框架中还不存在的层。
    在这些情况下，你必须构建自定义层。在本节中，我们将向你展示如何构建。
"""

"""
    不带参数的层
"""
class CenteredLayer(nn.Module):
    def __init__(self):
        super(CenteredLayer, self).__init__()
    def forward(self,X):
        #顶一个一个类，要从其输入中减去均值
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

" 也可以把层当做更为复杂的组件并合并到更复杂的模型中，跟WAlign一样 "
net = nn.Sequential(nn.Linear(8 ,128), CenteredLayer())

Y = net(torch.rand(4, 8))
print(Y.mean())



"""
    带参数的层
        这些函数提供一些基本的管理功能，比如管理访问、初始化、共享、保存和加载模型参数。
        这样做的好处之一就是：我们不需要为每个自定义层编写自定义的序列化程序。
"""
"""
    自定义版本的全连接层，一共需要两个参数，一个表示权重，一个表示偏执项，因此总共需要输入两个参数：
        in_units和units，分别表示输入数和输出数。
"""
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) +self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)

#可以使用自定义层直接执行前向传播计算
linear(torch.rand(2, 5))

#还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))