#!/usr/bin/ python2
# -*- coding: utf-8 -*-

"""
@File       :   train1_5fold.py
@Time       :   2023/4/18 8:44
@Author     :   Pshenhao
@Version    :   v0.0.1
@Contact    :   pshenhao@qq.com
@Desc       :   生而无畏

"""
import albumentations as A
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from numpy import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from dataset import LoadDataset
from model0_myNet import myUNET
from model1_UNet import UNET
from model2_SegNet import SegNet
from model3_ResUNet import ResUNet
from model4_DeepLabV1 import DeepLabV1
from model5_DeepLabV2 import DeepLabV2
from model6_DeepLabV3 import DeepLabV3
from utils1_save import load_checkpoint, save_checkpoint, save_predictions_as_imgs
from utils2_model import check_accuracy

# Hyper parameter etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLD = 5
BATCH_SIZE = 4
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_KEY = 0  # The Key of Model Selection
SELECT_MODEL = ["myUNet", "UNet", "SegNet", "ResUNet", "DeepLabV1", "DeepLabV2", "DeepLabV3"]
MODEL_FOLD = ["myUNet/", "UNet/", "SegNet/", "ResUNet/", "DeepLabV1/", "DeepLabV2/", "DeepLabV3"]
DATA_FOLD = "./data/data_lung/"
SAVE_FOLD = "./saved_images/" + MODEL_FOLD[MODEL_KEY]
RECODE_FOLD = "./record/" + MODEL_FOLD[MODEL_KEY]
TRAIN_IMG_DIR = DATA_FOLD + "train_images"
TRAIN_MASK_DIR = DATA_FOLD + "train_masks"
VAL_IMG_DIR = DATA_FOLD + "val_images"
VAL_MASK_DIR = DATA_FOLD + "val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler, epoch):
    print(f"----Epoch:{epoch}----")
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

    train_transforms = A.Compose(
        [A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), A.Rotate(limit=35, p=1.0), A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.1), A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
         ToTensorV2(), ], )
    val_transforms = A.Compose([A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                                ToTensorV2(), ], )

    k_fold = KFold(n_splits=K_FOLD, shuffle=True, random_state=1996)
    print('-' * 15)

    train_dataset = LoadDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transforms)
    val_dataset = LoadDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transforms)
    all_dataset = ConcatDataset([train_dataset, val_dataset])

    # K-Fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(k_fold.split(all_dataset)):
        print(f"FOLD:{fold}")
        print("-" * 15)

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                  sampler=train_subsampler)
        test_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 sampler=test_subsampler)

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

        if LOAD_MODEL:
            load_checkpoint(torch.load(RECODE_FOLD + "my_checkpoint_5f.pth.tar"), model)

        check_accuracy(test_loader, model, device=DEVICE)
        scaler = torch.cuda.amp.GradScaler()

        epoch_list = []
        acc_list = []
        dice_list = []
        jaccord_list = []
        precision_list = []
        recall_list = []
        f_score_list = []
        specificity_list = []
        auc_list = []
        tp_list = []
        tn_list = []
        fp_list = []
        fn_list = []

        for epoch in range(NUM_EPOCHS):
            epoch_list.append(epoch)
            train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch)

            # save model
            check_point = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            save_checkpoint(check_point, filename=RECODE_FOLD + "my_checkpoint_5f.pth.tar")

            acc, dice, jaccord, precision, recall, f_score, specificity, auc, tp, tn, fp, fn = check_accuracy(
                test_loader, model, device=DEVICE, train="train_5f")

            acc_list.append(acc)
            dice_list.append(dice)
            jaccord_list.append(jaccord)
            precision_list.append(precision)
            recall_list.append(recall)
            f_score_list.append(f_score)
            specificity_list.append(specificity)
            auc_list.append(auc)
            tp_list.append(tp)
            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)

            save_predictions_as_imgs(test_loader, model,
                                     folder=SAVE_FOLD + "pred_images_5f/" + "saved_images{}".format(str(fold)),
                                     device=DEVICE)

        plt.plot(epoch_list, jaccord_list, precision_list, recall_list)
        fig = plt.gcf()
        fig.savefig(r'./record/train_5f.png'.format())
        plt.show()

        print(f"Mean Accuracy:{mean(acc_list)}")
        print(f"Mean Jaccord:{mean(jaccord_list)}")
        print(f"Mean Dice:{mean(dice_list)}")
        print(f"Mean Precision:{mean(precision_list)}")
        print(f"Mean Recall:{mean(recall_list)}")
        print(f"Mean F1-score:{mean(f_score_list)}")
        print(f"Mean Specificity:{mean(specificity_list)}")
        print(f"Mean AUC:{mean(auc_list)}")


if __name__ == '__main__':
    main()
