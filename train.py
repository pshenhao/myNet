#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   train.py
@Time       :   2023/4/17 16:46
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import os

import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model0_myNet import myUNET
from model1_UNet import UNET
from model2_SegNet import SegNet
from model3_ResUNet import ResUNet
from model4_DeepLabV1 import DeepLabV1
from model5_DeepLabV2 import DeepLabV2
from model6_DeepLabV3 import DeepLabV3
from utils1_save import load_checkpoint, save_checkpoint, save_predictions_as_imgs
from utils2_model import get_loaders, check_accuracy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyper parameter etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_KEY = 0  # The Key of Model Selection
SELECT_MODEL = ["myUNet", "UNet", "SegNet", "ResUNet", "DeepLabV1", "DeepLabV2", "DeepLabV3"]
MODEL_FOLD = ["myUNet/", "UNet/", "SegNet/", "ResUNet/", "DeepLabV1/", "DeepLabV2/", "DeepLabV3"]
DATA_FOLD = "./data/data_cell/"
SAVE_FOLD = "./saved_images/" + MODEL_FOLD[MODEL_KEY]
RECODE_FOLD = "./record/" + MODEL_FOLD[MODEL_KEY]
TRAIN_IMG_DIR = DATA_FOLD + "train_images"
TRAIN_MASK_DIR = DATA_FOLD + "train_masks"
VAL_IMG_DIR = DATA_FOLD + "val_images"
VAL_MASK_DIR = DATA_FOLD + "val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update
        loop.set_postfix(loss=loss.item())


def main():
    print("==> Model : ", SELECT_MODEL[MODEL_KEY])
    print("==> R_PATH: ", RECODE_FOLD)
    print("==> S_PATH: ", SAVE_FOLD)

    train_transform = A.Compose(
        [A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), A.Rotate(limit=35, p=1.0), A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.1), A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, ),
         ToTensorV2(), ], )

    val_transform = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                               A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, ),
                               ToTensorV2(), ])

    #### 1. Select Model
    if SELECT_MODEL[MODEL_KEY] == "myUNet":
        model = myUNET(in_channels=3, out_channels=1).to(DEVICE)
    elif SELECT_MODEL[MODEL_KEY] == "UNet":
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif SELECT_MODEL[MODEL_KEY] == "SegNet":
        model = SegNet(in_channels=3, out_channels=1).to(DEVICE)
    elif SELECT_MODEL[MODEL_KEY] == "ResUNet":
        model = ResUNet().to(DEVICE)
    elif SELECT_MODEL[MODEL_KEY] == "DeepLabV1":
        model = DeepLabV1(num_classes=1, num_block=[3, 4, 23, 3]).to(DEVICE)
        model.eval()
    elif SELECT_MODEL[MODEL_KEY] == "DeepLabV2":
        model = DeepLabV2(num_classes=1, num_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]).to(DEVICE)
        model.eval()
    elif SELECT_MODEL[MODEL_KEY] == "DeepLabV3":
        model = DeepLabV3(num_classes=1, num_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18], multi_grids=[1, 2, 4],
                          output_stride=8).to(DEVICE)
        model.eval()

    #### 2. Loss Function
    loss_fn = nn.BCEWithLogitsLoss()

    #### 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE,
                                           train_transform, val_transform, NUM_WORKERS, PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load(RECODE_FOLD + "my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    epoch_list = []
    acc_list = []
    dice_list = []
    jaccord_list = []

    print(f"------------ Begin to Train ------------")
    for epoch in range(NUM_EPOCHS):
        print(f"---- Epoch:{epoch} ----")
        epoch_list.append(epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint, filename=RECODE_FOLD + "my_checkpoint.pth.tar")

        # check accuracy
        acc, dice, jaccord = check_accuracy(val_loader, model, device=DEVICE)

        acc_list.append(acc)
        dice_list.append(dice)
        jaccord_list.append(jaccord)

        save_predictions_as_imgs(val_loader, model, folder=SAVE_FOLD + "pred_images/", device=DEVICE)

    print(f"------------ End the Train ------------")
    # plot1: Jaccord
    plt.plot(epoch_list, jaccord_list)
    fig = plt.gcf()
    fig.savefig(RECODE_FOLD + r'Jaccord.png'.format())
    plt.show()

    # plot2: Dice
    plt.plot(epoch_list, dice_list)
    fig = plt.gcf()
    fig.savefig(RECODE_FOLD + r'Dice.png'.format())
    plt.show()

    # plot3: Acc
    plt.plot(epoch_list, acc_list)
    fig = plt.gcf()
    fig.savefig(RECODE_FOLD + r'Acc.png'.format())
    plt.show()

    print(f"Max Dice:{max(dice_list)}")
    print(f"Max Jaccord:{max(jaccord_list)}")
    print(f"Max Accuracy:{max(acc_list)}")


if __name__ == '__main__':
    main()
