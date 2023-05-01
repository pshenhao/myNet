#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   utils4_test.py
@Time       :   2023/4/25 16:21
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import torch
import torchvision


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()
