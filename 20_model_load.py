import torch
import torchvision
from torch import nn

#加载方式1:加载模型
model1 = torch.load("vgg16_model1.pth")

print(model1)

print("*"*100)

#加载方式2：加载参数，只有参数，没有模型
model2 = torch.load("vgg16_model2.pth")
print(model2)

#加载方式2：加载模型
vgg16 = torchvision.models.vgg16()
vgg16.load_state_dict(torch.load("vgg16_model2.pth"))
print(vgg16)


#对应陷阱1:在自定义模型时，必须在加载模型的文件中包含这个类

class base_nn(nn.Module):
    def __init__(self):
        super(base_nn, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x


model = torch.load("nn_method1.pth")
print(model)