#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   test2_model.py
@Time       :   2023/4/20 14:37
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


class DeepLabV1(nn.Sequential):
    def __init__(self, n_classes, n_blocks):
        super(DeepLabV1, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.module = self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("fc", nn.Conv2d(2048, n_classes, 1))
        self.add_module("output", F.interpolate(size=(512, 512), mode="bilinear", align_corners=False))


def test():
    model = DeepLabV1(n_classes=1, n_blocks=[3, 4, 23, 3])
    model.eval()
    image = torch.randn(4, 3, 512, 512)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)


if __name__ == "__main__":
    test()
