import torch
import torchvision
from PIL import Image
from model import nn_cifar10

img_path = "./imgs/dog.png"
img = Image.open(img_path)
img = img.convert("RGB")
print(img)
# img.show()


transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)
img = torch.reshape(img,[1,3,32,32])

model = torch.load("model_50.pth",map_location=torch.device('cpu'))
print(model)

model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1))
