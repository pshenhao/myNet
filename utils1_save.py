#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   utils1_save.py
@Time       :   2023/4/17 15:49
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import torch
import torchvision


# Save state
def save_checkpoint(state, filename="./record/my_checkpoint.pth.tar"):
    print("=> Saving CheckPoint!")
    torch.save(state, filename)


# Load model
def load_checkpoint(checkpoint, model):
    print("=> Loading CheckPoint!")
    model.load_state_dict(checkpoint["state_dict"])


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
