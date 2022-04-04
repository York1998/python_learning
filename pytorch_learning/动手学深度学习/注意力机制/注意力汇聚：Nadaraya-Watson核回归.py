# 作者：York
# 时间：2022/4/4 14:31
import torch
from torch import nn

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5) #排序后的