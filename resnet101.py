import torch
import torch.nn as nn
from torchvision import models


def resNet101(num_class):

    model = models.resnet101(False)
    pre_model = torch.load('/home/edison/workspace/zmvision-outdir/XISHU/resnet101-5d3b4d8f.pth')
    model.load_state_dict(pre_model)
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in, num_class)
    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()
    return model
def load_resNet101(path,num_class=18):
    model = models.resnet101(False)
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in, num_class)
    pre_model = torch.load(path)
    model.load_state_dict(pre_model)
    return model
