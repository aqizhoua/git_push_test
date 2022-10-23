import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([
    [1,-0.5],
    [-1,3]
])

input = torch.reshape(input,(-1,1,2,2)) #第一维是batchsize
print(input.shape)

class base_relu(nn.Module):
    def __init__(self):
        super(base_relu, self).__init__()
        self.relu1 = ReLU() #唯一的参数inplace bool类型，如果为True,输入参数直接改变；如果为False,输出改变，输入参数不变(默认，也建议)


    def forward(self,input):
        output = self.relu1(input)
        return output

class base_sigmoid(nn.Module):
    def __init__(self):
        super(base_sigmoid, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

relu = base_relu()
output = relu(input)
print(output)

print("*"*100)

sigmoid = base_sigmoid()
output = sigmoid(input)
print(output)


dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs,targets = data


    output = sigmoid(imgs)

    writer.add_images("input",imgs,step)
    writer.add_images("output",output,step)
    step += 1

writer.close()

