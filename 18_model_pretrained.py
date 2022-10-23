import torchvision
from torch import nn

# train_data = torchvision.datasets.ImageNet("./dataset",split="train",download=True,transform=torchvision.transforms.ToTensor)

vgg16_false = torchvision.models.vgg16()
# vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_false)
print("ok")

#在现有vgg网络模型基础上最后加一层，将1000分类的输出变为10输出

vgg16_false.add_module("add_linear",nn.Linear(1000,10))

print(vgg16_false)

#在现有vgg模型中间classifier上加一层
vgg16_false.classifier.add_module("add_linear",nn.Linear(1000,10))

print(vgg16_false)


#现有层的修改
vgg16_false.classifier[6] = nn.Linear(4096,10)

print(vgg16_false)
