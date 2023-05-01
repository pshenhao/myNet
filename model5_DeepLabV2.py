#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   model5_DeepLabV2.py
@Time       :   2023/4/19 16:45
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils2_model import _ResLayer, _Stem


class _ASPP(nn.Module):

    # Atrous spatial pyramid pooling(ASPP)
    def __init__(self, in_channels, out_channels, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module("c{}".format(i),
                            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rate, dilation=rate, bias=True))

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


# DeepLab V2: Dilated ResNet + ASPP
class DeepLabV2(nn.Module):
    def __init__(self, num_classes, num_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        channels = [64 * 2 ** p for p in range(6)]
        self.models = nn.ModuleList()

        self.models.append(_Stem(channels[0]))
        self.models.append(_ResLayer(num_blocks[0], channels[0], channels[2], 1, 1))
        self.models.append(_ResLayer(num_blocks[1], channels[2], channels[3], 2, 1))
        self.models.append(_ResLayer(num_blocks[2], channels[3], channels[4], 1, 2))
        self.models.append(_ResLayer(num_blocks[3], channels[4], channels[5], 1, 4))
        self.models.append(_ASPP(channels[5], num_classes, atrous_rates))

    def forward(self, x):
        _, _, H, W = x.shape
        for m in self.models:
            x = m(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        return x


def test():
    model = DeepLabV2(num_classes=1, num_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])

    print(model)

    x = torch.randn(4, 3, 512, 512)
    print(x.shape)

    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    test()
