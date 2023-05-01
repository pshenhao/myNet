#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   model6_DeepLabV3.py
@Time       :   2023/4/19 22:36
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils2_model import _ConvBnReLU, _ResLayer, _Stem


class _ImagePoll(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ImagePoll, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_channels, out_channels, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


# Atrous spatial pyramid pooling with image-level feature
class _ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(in_channels, out_channels, 1, 1, 0, 1))

        for i, rate in enumerate(rates):
            self.stages.add_module("c{}".format(i + 1),
                                   _ConvBnReLU(in_channels, out_channels, 3, 1, padding=rate, dilation=rate))
        self.stages.add_module("img_pool", _ImagePoll(in_channels, out_channels))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


# DeepLab V3: Dilated ResNet with multi-grid + improved ASPP
class DeepLabV3(nn.Module):
    def __init__(self, num_classes, num_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        channels = [64 * 2 ** p for p in range(6)]
        concat_channels = 256 * (len(atrous_rates) + 2)
        self.models = nn.ModuleList()

        self.models.append(_Stem(channels[0]))
        self.models.append(_ResLayer(num_blocks[0], channels[0], channels[2], s[0], d[0]))
        self.models.append(_ResLayer(num_blocks[1], channels[2], channels[3], s[1], d[1]))
        self.models.append(_ResLayer(num_blocks[2], channels[3], channels[4], s[2], d[2]))
        self.models.append(_ResLayer(num_blocks[3], channels[4], channels[5], s[3], d[3], multi_grids))

        self.models.append(_ASPP(channels[5], 256, atrous_rates))

        self.models.append(_ConvBnReLU(concat_channels, 256, 1, 1, 0, 1))
        self.models.append(nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(self, x):
        _, _, H, W = x.shape
        for model in self.models:
            x = model(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


def test():
    model = DeepLabV3(num_classes=1, num_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4],
                      output_stride=8, )
    print(model)

    x = torch.randn(4, 3, 512, 512)
    print(x.shape)

    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()
