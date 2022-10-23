from PIL import Image
from torchvision import transforms

import cv2

from torch.utils.tensorboard import SummaryWriter #可视化

img_path = "data/hymenoptera_data/train/ants_images/7759525_1363d24e88.jpg"
img = Image.open(img_path)

# img.show()

tensor_trans = transforms.ToTensor()
print(type(img))
tensor_img = tensor_trans(img) #Args:pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
#可以转换PIL Image(PIL库读取图片)或者numpy.ndarray类型(opencv库读取图片）
print(type(tensor_img))

print("*"*50)
img2 = cv2.imread(img_path)
print(type(img2))
tensor_img2 = tensor_trans(img2)
print(type(tensor_img2))


#输出：事件文件所在文件夹名
"""
<class 'PIL.JpegImagePlugin.JpegImageFile'>
<class 'torch.Tensor'>
**************************************************
<class 'numpy.ndarray'>
<class 'torch.Tensor'>
"""

#可视化
writer = SummaryWriter("logs") #logs为事件文件所在文件夹名

writer.add_image("Tensor_img",tensor_img2)

print(img2.shape)
writer.add_image("numpy_img",img2,dataformats="HWC")


writer.close()


