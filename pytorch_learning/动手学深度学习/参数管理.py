# 作者：York
# 时间：2022/4/1 15:03
import torch
"""
有时我们希望提取参数，以便在其他环境中复用它们， 将模型保存下来，以便它可以在其他软件中执行， 
或者为了获得科学的理解而进行检查。
"""
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)


#TODO
# 参数访问
" 当通过Sequential类定义模型时，可以通过索引来访问模型的任意层，这就像模型是一个列表一样" \
" 每层的参数都在其属性中。 "

print(net[2].state_dict())
# 此结果告诉我们一些事：首先这个全连接层包含两个参数，分别是该层的权重和偏置。两者都存储为浮点数（float32）
# 注意参数名称允许唯一标识每个参数，即使在包含数百层的网络中也是如此

"""
    注意，每个参数都表示为参数类得一个实例。要对参数执行任何操作之前，首先需要访问底层的数值。
"""
print(type(net[2].bias))
print(net[2].bias) #参数是复合对象，包含值、梯度和额外信息。这就是我们需要显示参数值的原因
print(net[2].bias.data)

"""
    如果想一次性访问所有的参数（对于更复杂的块），需要递归整个树来提取每个子块的参数。
"""
print(*[ (name, param.shape) for name, param in net[0].named_parameters()])
print(*[ (name,param.shape) for name, param in net.named_parameters() ])
"即这为我们提供了另外一种访问网络参数的形式"
print(net.state_dict()['2.bias'].data) #上面给的是 ('2.bias', torch.Size([1])，正好可以取出来

"""
    也可以从嵌套快收集参数(以后可以用到论文里面的)
"""
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}',block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)
"""
输出为
Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)
"""
"因为层是分层嵌套的，所以我们也可以通过嵌套列表索引一样访问它们。我们访问第一个主要的块中、第二个子块的的第一层的偏置项"
print(rgnet[0][1][0].bias.data)

"""
    初始化所有的参数为给定的常数
"""
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
"""
    我们还可以对某些块应用不同的初始化方法。 
    例如，下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
"""
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

#TODO
# 参数绑定

# 需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8),nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
#检查参数是否相同
print(net[2].weight.data[0]==net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
"""
 这个例子表明第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。
 因此，如果我们改变其中一个参数，另一个参数也会改变。
 你可能会思考：当参数绑定时，梯度会发生什么情况？ 
 答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）
 和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。
"""
"""
    共享参数通常可以节省内存，并在以下方面有特定的好处：
    
    对于图像识别中的CNN，共享参数使网络能够在图像中的任何地方而不是仅在某个区域中查找给定的功能。
    对于RNN，它在序列的各个时间步之间共享参数，因此可以很好地推广到不同序列长度的示例。
    对于自动编码器，编码器和解码器共享参数。在具有线性激活的单层自动编码器中，共享权重会在权重矩阵的不同隐藏层之间强制正交。

"""