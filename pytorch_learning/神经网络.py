# 作者：York
# 时间：2022/3/29 16:20
import torch
"""
    可以使用torch.nn来构造神经网络
    nn宝依赖于autograd宝来定义模型并对他们求导。一个nn.Module包含各个层合一个forward(input)方法，该方法返回output
    
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #因此卷积核大小事5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        pass
