import torchvision
import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

class base_linear(nn.Module):
    def __init__(self):
        super(base_linear, self).__init__()
        self.linear1 = Linear(196608,10) #两个参数；输入特征数，输出特征数

    def forward(self,input):
        output = self.linear1(input)
        return output

dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

linear = base_linear()

for data in dataloader:
    imgs,targets = data
    print(imgs.shape) #[64, 3, 32, 32]
    # output = torch.reshape(imgs,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape) #[1, 1, 1, 196608]

    output = linear(output)
    print(output.shape)
