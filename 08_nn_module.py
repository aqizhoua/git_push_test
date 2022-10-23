import torch
from torch import nn


class nn_base(nn.Module): #Base class for all neural network modules.
    def __init__(self):
        super(nn_base, self).__init__() #调用父类

    def forward(self,input): #用法和__call__类似，输入参数不需要使用forward函数名显示调用，实例化类后，将参数放实例里就行，如下面示例
        output = input +1
        return output


base = nn_base()
# print(base)
x = torch.Tensor([1,2,3,4])
output = base(x)
print(output)



