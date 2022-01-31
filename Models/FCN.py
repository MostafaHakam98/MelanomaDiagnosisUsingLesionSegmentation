from torch.nn.modules.activation import Softmax
import torch
import torch.nn as nn
from torchsummary import summary

def last_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 2, 1),
        nn.Upsample(size=(224,224), mode='bilinear', align_corners=True)
    )

class VGG16(nn.Module):

    def __init__(self, n_class,pretrained_net=None, is_sigmoid=True):
        super().__init__()

        self.features=pretrained_net.features
        self.last_block  = last_block(512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        if is_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        transfer_learn=self.features(x)
        out=self.last_block(transfer_learn)
        return self.final_activation(out)
    def summary(self):
        print(summary(self.cuda(),(3,224,224)))
