import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class base_sequential(nn.Module):
    def __init__(self):
        super(base_sequential, self).__init__()
        self.conv1 = Conv2d(3,32,5,padding=2) #（32+2*2-5）/1+1=32
        self.maxpool1 = MaxPool2d(2) #(32-2)/2+1=16
        self.conv2 = Conv2d(32,32,5,padding=2)  #(16+2*2-5)/1+1=16
        self.maxpool2 = MaxPool2d(2)    #(16-2)/2+1=8
        self.conv3 = Conv2d(32,64,5,padding=2)  #(8+2*2-5)/1+1=8
        self.maxpool3 = MaxPool2d(2) #(8-2)/2+1=4
        self.flatten = Flatten()    #Flatten()函数和torch.flatten()等价 全部展平为64*16=1024
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)

        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)



        )

    def forward(self,x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)

        x = self.model1(x)

        return x

base = base_sequential()
print(base)
input = torch.ones((64,3,32,32)) #输入必须为4个参数，第一个参数为batch_size.即一次处理图片的数量，第二个到第四个参数为channel,H,W
output = base(input)
print(output.shape)

print("*"*100)

writer = SummaryWriter("logs_5.13")
writer.add_graph(base,input) #add_graph函数：计算图
writer.close()

