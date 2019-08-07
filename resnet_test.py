import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from dataload_v2 import MyDataset
from resnet101 import resNet101,load_resNet101
import numpy as np




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_class = 18

resNet = load_resNet101("./models/latest.pth")
data_path = '/home/edison/workspace/zmvision-outdir/XISHU/test/'
batch_size = 8
test_dataset = MyDataset(data_path, augment=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

use_gpu = torch.cuda.is_available()
if use_gpu:
    resNet = resNet.cuda()

correct = 0
total = 0

for i, (img, label) in enumerate(test_loader):
    if use_gpu:
        img, label = img.cuda(), label.cuda()
    out = resNet(img)
    _, pred = torch.max(out.data, 1)
    total += label.size(0)
    correct += (pred == label).sum().item()
    print(i,"/" ,len(test_loader), 'Acc: %.3f%% (%d/%d)'
              % (100. * correct / total, correct, total))
