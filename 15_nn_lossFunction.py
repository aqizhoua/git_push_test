import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input = torch.tensor([1.0,2,3])
target = torch.tensor([1.0,2,1])

input = torch.reshape(input,(1,1,1,3)) #batch_size,channel,W,H
target = torch.reshape(target,(1,1,1,3))

#回归问题
#L1损失：L1Loss
#MSELoss:平方损失

loss = L1Loss(reduction="sum") #要先实例化，再传参，不然报错
loss = L1Loss() #默认为算均值mean
result = loss(input,target)
print(result)

print("*"*100)


loss_mse = MSELoss()
result_mse = loss_mse(input,target)
print(result_mse)


#分类问题
#交叉熵损失：CrossEntropy

input = torch.tensor([0.1,0.2,0.3])
target = torch.tensor([1])

print(input.shape)
input = torch.reshape(input,(1,3)) #N:batch_size C:number of classes

loss_cross = CrossEntropyLoss()
result = loss_cross(input,target) #-log(exp(0.2)/(exp(0.1)+exp(0.2)+exp(0.3)))
print(result)
