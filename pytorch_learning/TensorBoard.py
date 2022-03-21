# 作者：York
# 时间：2022/3/21 14:34
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar('y=x',2*i,i) # 第二个是y周，第三个是x轴 this is y=2x
writer.close()

