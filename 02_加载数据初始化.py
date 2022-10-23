from torch.utils.data import Dataset
from PIL import Image
import os



class MyData(Dataset):
    def __init__(self,root_path,label_path):
        self.root_path=root_path
        self.label_path=label_path
        self.path=os.path.join(self.root_path,self.label_path)
        self.img_path=os.listdir(self.path)


    def __getitem__(self,index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path,img_name)
        img = Image.open(img_item_path)
        label = self.label_path
        return img,label

    def __len__(self):
        return len(self.img_path)


root_dir = "data/hymenoptera_data/train"
ants_label_dir = "ants_images"
bees_label_dir = "bees_images"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
train_dataset = ants_dataset+bees_dataset




