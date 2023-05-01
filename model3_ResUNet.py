#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   model3_ResUNet.py
@Time       :   2023/4/18 22:22
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import torch
import torch.nn as nn


class batchnorm_relu(nn.Module):
    def __init__(self, in_channels):
        super(batchnorm_relu, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(x))


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residual_block, self).__init__()

        # Convolutional layer
        self.br1 = batchnorm_relu(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.br2 = batchnorm_relu(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)

        # Shortcut connection
        self.short_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.br1(inputs)
        x = self.conv1(x)
        x = self.br2(x)
        x = self.conv2(x)

        y = self.short_conv(inputs)
        return x + y


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)  # 双线性插值
        self.rb = residual_block(in_channels + out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        x = torch.cat((x, skip_connection), dim=1)
        return self.rb(x)


class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # Encoder.1
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        # Encoder.2 && Encoder.3
        self.rb21 = residual_block(64, 128, stride=2)
        self.rb22 = residual_block(128, 256, stride=2)

        # Bridge
        self.rb3 = residual_block(256, 512, stride=2)

        # Decoder
        self.db1 = decoder_block(512, 256)
        self.db2 = decoder_block(256, 128)
        self.db3 = decoder_block(128, 64)

        # Output
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        x = self.br1(self.conv11(inputs))
        x = self.conv12(x)
        y = self.conv13(inputs)
        skip_connection1 = x + y

        skip_connection2 = self.rb21(skip_connection1)
        skip_connection3 = self.rb22(skip_connection2)

        bridge = self.rb3(skip_connection3)

        d1 = self.db1(bridge, skip_connection3)
        d2 = self.db2(d1, skip_connection2)
        d3 = self.db3(d2, skip_connection1)

        output = self.output(d3)
        return output


def test():
    x = torch.randn(4, 3, 512, 512)
    model = ResUNet()
    preds = model(x)
    print(x.shape)
    print(preds.shape)


if __name__ == '__main__':
    test()
