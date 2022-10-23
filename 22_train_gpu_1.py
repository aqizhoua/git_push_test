import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.CIFAR10("../dataset",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#利用dataloader加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


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

#创建网络模型
model = nn_cifar10()

#调用gpu
if torch.cuda.is_available():
    model = model.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()
#调用gpu
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

#优化器
learning_rate = 1e-2 #0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)


#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10

#tensorboard画出损失函数的变化
writer = SummaryWriter("./logs")

start_time = time.time()
for i in range(epoch):
    print("-------第{}轮训练开始-------".format(i+1))

    #训练步骤开始
    model.train()   #只对特定层有影响，比如batch_normal,dropout,在此处设置不设置都行
    for data in train_dataloader:
        imgs,targets = data
        #对数据（输入、标注）调用gpu
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = model(imgs)

        # print("imgs.shape:", imgs.shape)    #64,3,32,32
        # print("outputs.shape:", outputs.shape)  #64*10

        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step +=1
        if total_train_step%100 ==0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数：{},Loss:{}".format(total_train_step,loss.item())) #.item()会把tensor数据类型变为数字，次数加不加都一样
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    #测试步骤开始
    model.eval()  # 只对特定层有影响，比如batch_normal,dropout,在此处设置不设置都行
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():   #在测试阶段，不需要也不能计算参数的梯度进行更新
        for data in test_dataloader:
            imgs,targets = data
            # 对数据（输入、标注）调用gpu
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            # print("imgs.shape:",imgs.shape)
            outputs = model(imgs)
            # print("outputs.shape:",outputs.shape)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()   #对于分类问题求正确率accuracy
            total_accuracy += accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    #模型保存
    torch.save(model,"model_{}.pth".format(i+1))
    print("模型已保存")



writer.close()






