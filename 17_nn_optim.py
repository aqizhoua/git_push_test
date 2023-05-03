#在之前写的神经网络中用Loss Function和优化器
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear, Flatten, CrossEntropyLoss
from torch.utils.data import DataLoader


class base_network(nn.Module):
    def __init__(self):
        super(base_network, self).__init__()
        self.model1 = Sequential(
        Conv2d(3, 32, 5, padding=2),
        MaxPool2d(2),
        Conv2d(32, 32, 5, padding=2),
        MaxPool2d(2),
        Conv2d(32, 64, 5, padding=2),
        MaxPool2d(2),
        Flatten(),
        Linear(1024, 64),
        Linear(64, 10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x


dataset = torchvision.datasets.CIFAR10(r"D:\data\pytorch_notes_review\dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

network = base_network()

loss_cross = CrossEntropyLoss()

optim = torch.optim.SGD(network.parameters(),lr=0.01) #一般就这两个最基本的参数


for epoch in range(20):
    running_loss = 0
    for data in dataloader:
        imgs,targets = data
        outputs = network(imgs)
        result_loss = loss_cross(outputs,targets)
        optim.zero_grad() #梯度清零
        result_loss.backward() #反向传播，即计算梯度
        optim.step() #更新参数
        running_loss += result_loss
    print(running_loss)





