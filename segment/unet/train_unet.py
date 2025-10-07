import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from utils import FocalDiceLoss  # 自定义 loss，确保适配 sigmoid 输出
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for segmentation")
    parser.add_argument("--image_root", type=str, required=True, help="图像根目录")
    parser.add_argument("--mask_root", type=str, required=True, help="掩码根目录")
    parser.add_argument("--excel_dir", type=str, required=True, help="五折Excel目录，包含 fold{n}_train.xlsx")
    parser.add_argument("--image_size", type=int, default=256, help="训练时的输入尺寸")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--save_dir", type=str, default="segment/weights", help="模型保存目录")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--fold_start", type=int, default=1, help="开始折编号（含）")
    parser.add_argument("--fold_end", type=int, default=5, help="结束折编号（含）")
    return parser.parse_args()

# 1通道UNet
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features * 8 * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 4 * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 2 * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(OrderedDict([
            (name + "_conv1", nn.Conv2d(in_channels, features, 3, padding=1, bias=False)),
            (name + "_bn1", nn.BatchNorm2d(features)),
            (name + "_relu1", nn.ReLU(inplace=True)),
            (name + "_conv2", nn.Conv2d(features, features, 3, padding=1, bias=False)),
            (name + "_bn2", nn.BatchNorm2d(features)),
            (name + "_relu2", nn.ReLU(inplace=True)),
        ]))

class LungUltrasoundDataset(Dataset):
    def __init__(self, df, image_root, mask_root, image_size=256):
        self.df = df
        self.image_root = image_root
        self.mask_root = mask_root
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row["检查流水号"])

        image_path = os.path.join(self.image_root, uid)
        mask_path = os.path.join(self.mask_root, uid)

        image_file = next(f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        mask_file = next(f for f in os.listdir(mask_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

        image = cv2.imread(os.path.join(image_path, image_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.image_size, self.image_size)).astype(np.float32)
        mask = (mask > 127).astype(np.float32)

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        return torch.tensor(image), torch.tensor(mask)

def train(fold, args):
    train_df = pd.read_excel(os.path.join(args.excel_dir, f"fold{fold}_train.xlsx"))

    train_dataset = LungUltrasoundDataset(train_df, args.image_root, args.mask_root, args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1} - Train"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Fold {fold} Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"unet_fold{fold}.pth"))
    print(f"✅ Fold {fold} 训练完成，模型已保存。")

if __name__ == "__main__":
    args = parse_args()
    for fold in range(args.fold_start, args.fold_end + 1):
        train(fold, args)
