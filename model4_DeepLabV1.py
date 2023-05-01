#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   model4_DeepLabV1.py
@Time       :   2023/4/19 9:13
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


class DeepLabV1(nn.Module):
    def __init__(self, num_classes, num_block):
        super(DeepLabV1, self).__init__()
        channels = [64 * 2 ** p for p in range(6)]

        self.module = nn.ModuleList()

        self.module.append(_Stem(channels[0]))
        self.module.append(_ResLayer(num_block[0], channels[0], channels[2], 1, 1))
        self.module.append(_ResLayer(num_block[1], channels[2], channels[3], 2, 1))
        self.module.append(_ResLayer(num_block[2], channels[3], channels[4], 1, 2))
        self.module.append(_ResLayer(num_block[3], channels[4], channels[5], 1, 4))
        self.module.append(nn.Conv2d(2048, num_classes, 1))

    def forward(self, x):
        _, _, H, W = x.shape
        for model in self.module:
            x = model(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)  # 双线性插值

        return x


def test():
    model = DeepLabV1(num_classes=1, num_block=[3, 4, 23, 3])
    print(model)

    x = torch.randn(4, 3, 512, 512)
    print(x.shape)

    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    test()
