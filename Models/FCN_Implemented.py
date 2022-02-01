import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def triple_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


def last_block(in_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 2, 1),
        nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
    )


class VGG16_New(nn.Module):

    def __init__(self, n_class, is_sigmoid=True):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = triple_conv(128, 256)
        self.dconv_down4 = triple_conv(256, 512)
        self.dconv_down5 = triple_conv(512, 512)
        self.last_block = last_block(512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if is_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)

        x = self.maxpool(conv4)

        conv5 = self.dconv_down5(x)

        x = self.maxpool(conv5)
        out = self.last_block(x)
        return self.final_activation(out)