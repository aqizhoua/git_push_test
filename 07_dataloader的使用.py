import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

test_dataset = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor(),)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True) #shuffle是否打乱决定了后面每个epoch是否一样

print("*"*100)
print(test_dataset[0])
img,target = test_dataset[0]
print(img)
print(img.shape)
print(target)
print("*"*100)

writer = SummaryWriter("logs")

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        print(imgs.shape) #torch.Size([64,3,32,32]) batch_size channel height width
        print(targets)
        writer.add_images("epoch {}".format(epoch),imgs,step)
        step += 1


writer.close()