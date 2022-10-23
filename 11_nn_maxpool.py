import torch

import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor(
    [[1.0,2,0,3,1],
     [0,1,2,3,1],
     [1,2,1,0,0],
     [5,2,3,1,1],
     [2,1,0,1,1]]
)

input = torch.reshape(input,(-1,1,5,5))

print(input.shape) #4个维度分别为batch_size,channel,H,W


class maxpool_base(nn.Module):
    def __init__(self):
        super(maxpool_base, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True) #ceil_mode为True代表向上去整，即当剩余尺寸小于kernel_size大小时，仍然进行处理

    def forward(self,input):
        output = self.maxpool1(input)#(5-3)/3+1=1 stride默认= kernel_size
        return output

base = maxpool_base()
output = base(input)
print(output)

print("*"*100)

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter("logs")

step=0
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)
    output = base(imgs)
    writer.add_images("output",output,step)
    step += 1

writer.close()





