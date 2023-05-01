#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   model0_myNet.py
@Time       :   2023/4/20 14:26
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from utils2_model import get_kernel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),

                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), )

    def forward(self, x):
        return self.conv(x)


# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channels = in_channels // reduction
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(nn.Linear(in_features=in_channels, out_features=mid_channels), nn.ReLU(inplace=True),
                                 nn.Linear(in_features=mid_channels, out_features=in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pooling(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        max_out = self.mlp(self.max_pooling(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avg_out + max_out)


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x) * x


# Attention Block
class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.channel = ChannelAttention(in_channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        return torch.add(self.channel(x), self.spatial(x))


# Inception Block
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1))
        self.branch2 = nn.Sequential(nn.AvgPool2d(kernel_size=2),
                                     nn.ConvTranspose2d(in_channels, 128, kernel_size=2, stride=2))
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1),
                                     nn.Conv2d(128, 64, kernel_size=3, padding=1))

        self.branch3_1 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.branch3_2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

        self.branch4_1 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.branch4_2 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x3_1 = self.branch3_1(x3)
        x3_2 = self.branch3_2(x3)
        x4_1 = self.branch4_1(x4)
        x4_2 = self.branch4_2(x4)

        return torch.cat((x1, x2, x3_1, x3_2, x4_1, x4_2), dim=1)


class myUNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(myUNET, self).__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # Attention part of myNet
        for feature in reversed(features):
            self.attention.append(Attention(feature))

        # Batch Normalization of Outline
        for feature in features:
            self.batchnorm.append(
                nn.BatchNorm2d(feature, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True))

        self.inception = Inception(in_channels)
        self.batchnorm_relu = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Decoder part
        for idx, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)

            ######### Add OutLine #########
            _, kernel, _ = get_kernel()
            kernel = torch.FloatTensor(kernel).expand(len(x[0]), len(x[0]), 3, 3).to(device=DEVICE)
            weight = torch.nn.Parameter(data=kernel, requires_grad=False).to(device=DEVICE)
            y = torch.nn.functional.conv2d(x, weight, padding=1).to(device=DEVICE)
            x = self.batchnorm[idx](x) + self.batchnorm[idx](y)
            ######### =========== #########

            x = self.pool(x)

        skip_connections = skip_connections[::-1]

        ######### Add Inception #########
        y = self.inception(x)
        x = self.batchnorm_relu(x)
        y = self.batchnorm_relu(y)
        x = torch.cat((x, y), dim=1)
        ######### ============= #########

        # Encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shpe[2:])

            ######### Add Attention #########
            y = self.attention[idx // 2](skip_connection)
            concat_skip = torch.cat((y, x), dim=1)
            ######### ============= #########

            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class testModel(nn.Module):
    def __init__(self, in_channels):
        super(testModel, self).__init__()
        self.attention = Attention(in_channels)

    def forward(self, x):
        return self.attention(x)


def test_m():
    model = testModel(in_channels=256)
    print(model)

    x = torch.randn(4, 256, 128, 128)
    print(x.shape)

    y = model(x)
    print(y.shape)


def test():
    model = myUNET(in_channels=3, out_channels=1)
    print(model)

    x = torch.randn(4, 3, 512, 512)
    print(x.shape)

    preds = model(x)
    print(preds.shape)


if __name__ == '__main__':
    test()
