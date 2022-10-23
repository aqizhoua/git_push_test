import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

#train=False 返回的是测试数据集
dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class nn_base(nn.Module):
    def __init__(self):
        super(nn_base, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0) #padding不为0时，补全的数也是0

    def forward(self,x):
        x = self.conv1(x)
        return x


base = nn_base()
print(base)
# nn_base(
#   (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
# )

step=0
for data in dataloader:
    img,target = data


    output = base(img)
    print(output.shape)
    print(img.shape)

    #上两行输出如下 4个参数依次为：batch_size,channel_numbers,width,height
    # torch.Size([64, 6, 30, 30])
    # torch.Size([64, 3, 32, 32])

    #可视化

#   torch.Size([64, 3, 32, 32])
    writer.add_images("input",img,step)


#   torch.Size([64, 6, 30, 30])
    output = torch.reshape(output,(-1,3,30,30)) #因为tensorboard最多只能显示三通道，所以强行从6通道转为3通道，失去了意义

    writer.add_images("output",output,step)

    step += 1
