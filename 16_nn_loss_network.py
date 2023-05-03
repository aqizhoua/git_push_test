#在之前写的神经网络中用Loss Function
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

for data in dataloader:
    imgs,targets = data
    outputs = network(imgs)
    # print(outputs)
    # print(targets)

    result = loss_cross(outputs,targets)
    print(result)
    result.backward() #反向传播
    print("ok")






