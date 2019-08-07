import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from dataload_v2 import MyDataset
from resnet101 import resNet101
import numpy as np


num_class = 18

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
resNet = resNet101(num_class)
data_path = '/home/edison/workspace/zmvision-outdir/XISHU/train/'
num_epochs = 30

learning_rate = 0.1

batch_size = 8
train_dataset = MyDataset(data_path, augment=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
start_lr = learning_rate/100
warmup_iters = int(9000/batch_size/3)
add_lr = (learning_rate-start_lr)/warmup_iters
step = [10,20,25]
use_gpu = torch.cuda.is_available()
if use_gpu:
    resNet = resNet.cuda()
optimizer = torch.optim.SGD(resNet.parameters(), lr=start_lr, momentum=0.9, weight_decay=0.001)

resNet.train()
batch_num = 0
for epoch in range(num_epochs):
    test_loss = 0
    correct = 0
    total = 0
    pre_class = np.zeros((1, 18))
    true_class = np.zeros((1, 18))
    for i, (img, label) in enumerate(train_loader):
        for j in label:
            true_class[0,j]+=1
        if use_gpu:
            img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        out = resNet(img)

        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        test_loss += loss.item()
        _, pred = torch.max(out.data, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()
        for j in range(batch_size):
            if pred[j] == label[j]:
                pre_class[0, pred[j]] += 1
        print("epoch:",epoch," ",i,"/" ,len(train_loader), 'learning Rate: %.4f | Loss: %.4f | Acc: %.3f%% (%d/%d)'
              % (start_lr, test_loss / (i + 1), 100. * correct / total, correct, total))
        if batch_num < warmup_iters:
            start_lr+=add_lr
            optimizer = torch.optim.SGD(resNet.parameters(), lr=start_lr)
        batch_num += 1
    acc_class = pre_class/true_class
    print("pre_class ---- ",pre_class)
    print("true_class ---- ",true_class)
    print("acc_class ---- ",acc_class)
    if (epoch+1) in step:
        start_lr =start_lr/10
        optimizer = torch.optim.SGD(resNet.parameters(), lr=start_lr)
    torch.save(resNet.state_dict(), "./models/epoch_"+str(epoch+1)+".pth")
torch.save(resNet.state_dict(), "./models/latest.pth")
