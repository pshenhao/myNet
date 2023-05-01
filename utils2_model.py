#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   utils2_model.py
@Time       :   2023/4/17 22:54
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader

from dataset import LoadDataset


# Load Dataset
def get_loaders(train_dir, train_mask_dir, val_dir, val_mask_dir, batch_size, train_transform, val_transform,
                num_workers=4, pin_memory=True):
    train_ds = LoadDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=True)
    val_ds = LoadDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                            shuffle=False)
    return train_loader, val_loader


# Compute Acc
def check_accuracy(loader, model, device="cuda", train="train"):
    num_correct = 0
    num_pixels = 0
    tp = fp = tn = fn = 0
    auc = tpr = fpr = 0
    specificity = 0
    f_score = 0
    dice_score = 0
    jaccord_score = 0
    recall = 0
    precision = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            x = model(x)  # 在此处计算
            x = torch.tensor(x)
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            tp += (preds * y).sum()
            tn += num_correct - tp
            fp += (preds - preds * y).sum()
            fn += (y - preds * y).sum()
            tpr += tp / (tp + fn)
            fpr += fp / (fp + tn)

            a = y.cpu().numpy()
            b = preds.cpu().numpy()
            aa = list(np.array(a).flatten())
            bb = list(np.array(b).flatten())
            auc = metrics.roc_auc_score(aa, bb)
            precision += tp / ((tp + fp) + 1e-8)
            recall += tp / ((tp + fn) + 1e-8)
            f_score += (2 * tp) / ((fp + 2 * tp + fn) + 1e-8)
            specificity += tn / ((tn + fp) + 1e-8)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            jaccord_score += (preds * y).sum() / ((preds + y).sum() + 1e-8 - (preds * y).sum())

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.4f}")
    print(f"Dice score: {dice_score / len(loader)}")
    print(f"Jaccord score: {jaccord_score / len(loader)}")
    print(f"Precision:{precision / len(loader)}")
    print(f"Recall:{recall / len(loader)}")
    print(f"F1-score:{f_score / len(loader)}")
    print(f"Specificity:{specificity / len(loader)}")
    print(f"AUC:{auc}")
    print(f"TP total:{tp}")
    print(f"TN total:{tn}")
    print(f"FP total:{fp}")
    print(f"FN total:{fn}")

    model.train()

    if train == "train_5f":
        return (num_correct / num_pixels * 100).cpu().numpy(), (dice_score / len(loader)).cpu().numpy(), (
                jaccord_score / len(loader)).cpu().numpy(), (precision / len(loader)).cpu().numpy(), (
                       recall / len(loader)).cpu().numpy(), (f_score / len(loader)).cpu().numpy(), (specificity / len(
            loader)).cpu().numpy(), auc, tp.cpu().numpy(), tn.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()
    else:
        return (num_correct / num_pixels * 100).cpu().numpy(), (dice_score / len(loader)).cpu().numpy(), (
                jaccord_score / len(loader)).cpu().numpy()


############### UNET ################
def get_kernel():
    ### See in https://setosa.io/ev/image-kernels/

    # k1: blur
    k1 = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])

    # k2:outline
    k2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # k3:sharpen
    k3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    return k1, k2, k3


def build_sharp_blocks(layer):
    in_channels = layer.shape[1]
    _, w, _ = get_kernel()
    w = np.expand_dims(w, axis=0)
    w = np.repeat(w, in_channels, axis=0)
    w = np.expand_dims(w, axis=0)

    return torch.FloatTensor(w)


# Compute Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "The size of predict and target must be equal!"
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        inter_section = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (inter_section + self.epsilon) / (union + self.epsilon)
        return score


class _ConvBnReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU, self).__init__()
        self.add_module("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=dilation, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.001))

        if relu:
            self.add_module("relu", nn.ReLU())


_BOTTLENECK_EXPANSION = 4


class _Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, down_sample):
        super(_Bottleneck, self).__init__()
        mid_channels = out_channels // _BOTTLENECK_EXPANSION
        self.reduce = _ConvBnReLU(in_channels, mid_channels, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_channels, mid_channels, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_channels, out_channels, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_channels, out_channels, 1, stride, 0, 1, False) if down_sample else nn.Identity())

    def forward(self, x):
        y = self.reduce(x)
        y = self.conv3x3(y)
        y = self.increase(y)
        y += self.shortcut(x)
        return F.relu(y)


# Residual layer
class _ResLayer(nn.Sequential):
    def __init__(self, num_layers, in_channels, out_channels, stride, dilation, multi_grids=None):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(num_layers)]
        else:
            assert num_layers == len(multi_grids)

        for i in range(num_layers):
            self.add_module("block{}".format(i + 1), _Bottleneck(in_channels=(in_channels if i == 0 else out_channels),
                                                                 out_channels=out_channels,
                                                                 stride=(stride if i == 0 else 1),
                                                                 dilation=dilation * multi_grids[i],
                                                                 down_sample=(i == 0)))


class _Stem(nn.Sequential):
    def __init__(self, out_channels):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, out_channels, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class ResNet(nn.Sequential):
    def __init__(self, num_classes, num_blocks):
        super(ResNet, self).__init__()
        channels = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(channels[0]))
        self.add_module("layer2", _ResLayer(num_blocks[0], channels[0], channels[2], 1, 1))
        self.add_module("layer3", _ResLayer(num_blocks[1], channels[2], channels[3], 2, 1))
        self.add_module("layer4", _ResLayer(num_blocks[2], channels[3], channels[4], 2, 1))
        self.add_module("layer5", _ResLayer(num_blocks[3], channels[4], channels[5], 2, 1))
        self.add_module("pool5", nn.AdaptiveAvgPool2d(1))
        self.add_module("flatten", nn.Flatten())
        self.add_module("fc", nn.Linear(channels[5], num_classes))


def test():
    model = ResNet(num_classes=1, num_blocks=[3, 4, 23, 3])
    x = torch.rand(4, 3, 512, 512)
    y = model(x)
    print(model)
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    test()
