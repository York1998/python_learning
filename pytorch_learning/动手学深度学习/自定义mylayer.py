# 作者：York
# 时间：2022/4/1 11:19

import torch
import torch.nn as nn
from torch.nn import functional as F
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


class Mylayer(torch.nn.Module):
    '''
    实现 y = weights*sqrt(x2+bias)
    bias 的维度是（in_fearures,)，注意这里为什么是in_features,而不是out_features，注意体会这里和Linear层的区别所在
    weights 的维度是（in_features, out_features）注意这里为什么是（in_features, out_features）,而不是（out_features, in_features），注意体会这里和Linear层的区别所在
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(Mylayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        input_ = torch.pow(input, 2) + self.bias
        y = torch.matmul(input_, self.weight)
        return y

N, D_in, D_out = 10, 5, 3 #一共10组样本，输入特征为5，输出特征为3

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.mylayer1 = Mylayer(D_in, D_out) #自定义的

    def forward(self, x):
        x = self.mylayer1(x)
        return x

model = MyNet()
print(model)

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
for t in range(10):
    #第一步，数据的前向传播，计算预测值p_pred
    y_pred = model(x)
    #第二步，计算预测值p_pred与真实值之间的误差
    loss = loss_fn(y_pred, y)
    print(f"第 {t}个epoch, 损失是{loss.item()}")

    #在反向传播之前，将模型的梯度归零
    optimizer.zero_grad()

    #第三步，反向传播误差
    loss.backward()
    #直接通过梯度一步到位，更新完整个网络的训练参数
    optimizer.step()
