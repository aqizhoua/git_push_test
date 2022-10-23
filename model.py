import torch
from torch import nn

#搭建神经网络
class nn_cifar10(nn.Module):
    def __init__(self):
        super(nn_cifar10, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2), #（32+2*2-5）/1+1=32
            nn.MaxPool2d(2), #(32-2)/2+1=16
            nn.Conv2d(32,32,5,padding=2), #(32-5+2*2)/1+1=32
            nn.MaxPool2d(2), #(16-2)/2+1=8
            nn.Conv2d(32,64,5,padding=2), #(8-5+2*2)/1+1=8
            nn.MaxPool2d(2), #(8-2)/2+1=4
            nn.Flatten(), #展平 64*4*4=1024
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    nn = nn_cifar10()
    input = torch.ones((64,3,32,32))
    output = nn(input)
    print(output.shape)

