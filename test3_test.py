#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   test3_test.py
@Time       :   2023/4/25 16:00
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""

import os

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset import LoadDataset
from model0_myNet import myUNET
from model1_UNet import UNET
from model2_SegNet import SegNet
from model3_ResUNet import ResUNet
from model4_DeepLabV1 import DeepLabV1
from model5_DeepLabV2 import DeepLabV2
from model6_DeepLabV3 import DeepLabV3
from utils1_save import load_checkpoint
from utils4_test import save_predictions_as_imgs

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyper parameter etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 30
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
MODEL_KEY = 0  # The Key of Model Selection
SELECT_MODEL = ["myUNet", "UNet", "SegNet", "ResUNet", "DeepLabV1", "DeepLabV2", "DeepLabV3"]
MODEL_FOLD = ["myUNet/", "UNet/", "SegNet/", "ResUNet/", "DeepLabV1/", "DeepLabV2/", "DeepLabV3"]
DATA_FOLD = "./data/"
SAVE_FOLD = "./saved_images/" + "test/"
TEST_FOLD = DATA_FOLD + "test/test/"
TEST_MASK = DATA_FOLD + "test/mask/"
RECODE_FOLD = "./record/" + MODEL_FOLD[MODEL_KEY]


def main():
    print("==> Model : ", SELECT_MODEL[MODEL_KEY])
    print("==> S_PATH: ", SAVE_FOLD)

    test_loader = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                             A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0, ),
                             ToTensorV2(), ])

    test_ds = LoadDataset(image_dir=TEST_FOLD, mask_dir=TEST_MASK)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                             shuffle=False)

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

    load_checkpoint(torch.load(RECODE_FOLD + "my_checkpoint.pth.tar", map_location=torch.device('cpu')), model)

    # check accuracy
    save_predictions_as_imgs(test_loader, model, device=DEVICE)


if __name__ == '__main__':
    main()
