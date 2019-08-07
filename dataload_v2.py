import os, cv2
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms.transforms as transforms
from PIL import Image



# def get_img_paths(directory):
#     return [x.path for x in os.scandir(directory) if x.name.endwith(".jpg") or x.name.endwith(".png")]

def read_files(file_path, file_type='.json'):
    file_list = os.listdir(file_path)
    file_list.sort()
    file_list = [file for file in file_list if file.endswith(file_type)]
    label_list = [int(file.split('_')[0]) for file in file_list if file.endswith(file_type)]
    return file_list, label_list



class MyDataset(Dataset):
    def __init__(self, root, augment=None):
        self.image_files, self.label_list = read_files(root, '.png')
        self.augment = augment   
        self.root = root
    def __getitem__(self, index):
        image = Image.open(self.root + self.image_files[index])
        if self.augment:
            image = self.augment(image) 
        # image.resize_(3,224,224)
        image = image/255
        label = self.label_list[index]
        return image, label
    def __len__(self):
        return len(self.image_files)





