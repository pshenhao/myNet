#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   utils3_module.py
@Time       :   2023/4/21 8:44
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),

                                  nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), )

    def forward(self, x):
        return self.conv(x)


# InceptionBlock.1
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 1x1(32)
        self.branch1x1 = torch.nn.Conv2d(in_channels, 40, kernel_size=1)  # in_channels -> 32

        # 1x1(16) -> 5x5(72)
        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 72, kernel_size=5, padding=2)  # in_channels -> 72

        # 1x1(16) -> 3x3(24) -> 3x3(80)
        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24, 72, kernel_size=3, padding=1)  # in_channels -> 80

        # AvePooling -> 1x1(72)
        self.branch_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.p_branch1x1 = torch.nn.Conv2d(in_channels, 72, kernel_size=1)  # in_channels -> 72

    def forward(self, x):
        x_branch1x1 = self.branch1x1(x)

        x_branch5x5 = self.branch5x5_1(x)
        x_branch5x5 = self.branch5x5_2(x_branch5x5)

        x_branch3x3 = self.branch3x3_1(x)
        x_branch3x3 = self.branch3x3_2(x_branch3x3)
        x_branch3x3 = self.branch3x3_3(x_branch3x3)

        x_avg_branch1x1 = self.branch_pool(x)
        x_avg_branch1x1 = self.p_branch1x1(x_avg_branch1x1)

        outputs = [x_branch1x1, x_branch5x5, x_branch3x3, x_avg_branch1x1]
        outputs = torch.cat(outputs, dim=1)

        return outputs


# InceptionBlock.2
class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, 2, kernel_size=1),
                                     nn.Conv2d(2, 4, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.Conv2d(4, 8, kernel_size=(7, 1), padding=(3, 0)),
                                     nn.Conv2d(8, 16, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.Conv2d(16, 32, kernel_size=(7, 1), padding=(3, 0)))
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=1),
                                     nn.Conv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
                                     nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)))
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=2), nn.ConvTranspose2d(3, 64, kernel_size=2, stride=2))
        self.branch4 = nn.Sequential(nn.Conv2d(in_channels, 96, kernel_size=1))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        return torch.cat((x1, x2, x3, x4), dim=1)


class BottleInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleInception, self).__init__()
        self.Inception1 = DoubleConv(in_channels, out_channels)
        self.Inception2 = InceptionA(in_channels)
        self.Inception3 = InceptionB(in_channels)

        self.batchNorm_relu = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.batchNorm_relu(self.Inception1(x))
        x2 = self.batchNorm_relu(self.Inception2(x))
        x3 = self.batchNorm_relu(self.Inception3(x))

        return torch.cat((x1, x2, x3), dim=1)
