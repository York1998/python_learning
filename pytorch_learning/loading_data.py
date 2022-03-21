# 作者：York
# 时间：2022/3/21 12:33
"""
    在pytorch中加载数据主要设计两个类：
        Dataset：提供一种方式去获取数据及其label
            Dataset类是Pytorch中图像数据集中最为重要的一个类，也是Pytorch中所有数据集加载类中应该继承的父类
            。其中父类中的两个私有成员函数必须被重载：
            （
                def __getitem__(self,index)：应该编写支持数据集索引的函数，__getitem__接收一个index，
                    然后返回图片数据和标签，这个index通常指的是list的index，这个list的每个元素就包含了图片
                    数据的路径和标签信息。
                如何制作这个list？：通常做法是将图片路径和路径的标签存储在一个txt中，然后从该txt中读取。
                def len(self)：应该返回数据集的大小
            ）
        Dataloader：为后面的网络提供不同的数据形式
"""
from torch.utils.data import Dataset

class FirstDataset(Dataset):

    def __init__(self):
        #TODO
        # 1、初始化文件路径和文件名列表
        # 也就是在这个模块里，所做的工作就是初始化该类别的一些基本参数
        pass

    def __getitem__(self, idx):
        #TODO
        # 1、从文件中读取一个数据，例如：使用numpy.fromfile
        # 2、预处理数据
        # 3、返回数据对
        # 第一步read one data是一个data
        pass
    def __len__(self):
        #TODO(york):应该将0改为数据集的总大小
        pass