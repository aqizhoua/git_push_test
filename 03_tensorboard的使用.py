from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

#PIL python image library

writer = SummaryWriter("logs")
image_path = "data/hymenoptera_data/train/bees_images/39747887_42df2855ee.jpg"
img_PIL = Image.open(image_path)
img_PIL.show()
img_array=np.array(img_PIL)

#add_image方法只能读取img_tensor (torch.Tensor, numpy.ndarray, or string/blobname)这些数据类型，所以要把PILimage转成numpy数组
writer.add_image("train",img_array,1,dataformats="HWC") #dataformats如果三个维度与默认的不同，需要指定 H 高度 W 宽度 C channel 通道数
# y=2x



for i in range(100):
    writer.add_scalar("y=x",i,i) #第一个参数为tag，即标题 第二个参数为y轴，第三个参数为x轴

for i in range(100):
    writer.add_scalar("y=x^2",i**2,i)

writer.close()