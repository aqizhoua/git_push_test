import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16()
#保存方式1,保存模型结构+模型参数
torch.save(vgg16,"vgg16_model1.pth")

#保存方式2，保存模型参数（官方推荐），会小一丢丢
torch.save(vgg16.state_dict(),"vgg16_model2.pth")


#陷阱
class base_nn(nn.Module):
    def __init__(self):
        super(base_nn, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

base = base_nn()
torch.save(base,"nn_method1.pth")

