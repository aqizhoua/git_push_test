from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img = Image.open("data/pytorch.jpeg")
print(img)


writer = SummaryWriter("logs")

#totensor()
trans_img = transforms.ToTensor()
img_tensor = trans_img(img)

writer.add_image("totensor",img_tensor)

print("*"*100)

#normalize 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]) #两个列表分别代表均值和标准差 ，每个列表三个参数是因为图片是RGB三通道的 ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("tonorm",img_norm)

#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#img PIL -> resize -> img_resize PIL
img_resize = trans_resize(img)
#img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_img(img_resize)

writer.add_image("Resize",img_resize,0)
print(type(img_resize))


#compose - resize - 2  compose的作用是：把多个步骤整合到一起
trans_resize2 = transforms.Resize(512) #只输入一个参数，则不改变长宽比
#PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize2,trans_img])
img_resize2 = trans_compose(img)
writer.add_image("Resize",img_resize2,1)

#randomCrop 随机裁剪
trans_random = transforms.RandomCrop(50) #裁剪出的是正方形
trans_random = transforms.RandomCrop(50,40) #裁剪矩形
trans_compose2 = transforms.Compose([trans_random,trans_img])
for i in range(10): #随机裁剪，所以每次结果都不一样
    img_crop = trans_compose2(img)
    writer.add_image("RandomCropHW",img_crop,i)






writer.close()
