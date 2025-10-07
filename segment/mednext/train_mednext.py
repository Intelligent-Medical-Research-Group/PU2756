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
from mednextv1.MedNextV1 import MedNeXt
from sklearn.metrics import jaccard_score, f1_score
from utils import FocalDiceLoss
from scipy.ndimage import binary_fill_holes

def parse_args():
    parser = argparse.ArgumentParser(description="Train MedNeXt for segmentation")
    parser.add_argument("--image_root", type=str, required=True, help="图像根目录")
    parser.add_argument("--mask_root", type=str, required=True, help="掩码根目录")
    parser.add_argument("--excel_dir", type=str, required=True, help="五折Excel目录")
    parser.add_argument("--image_size", type=int, default=256, help="输入尺寸")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数")
    parser.add_argument("--in_channels", type=int, default=1, help="输入通道数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--pretrained", type=str, default="weight/mednext_pretrain_final.pth", help="预训练权重路径，可为空")
    parser.add_argument("--save_dir", type=str, default="segment/weights", help="模型保存目录")
    parser.add_argument("--fold_start", type=int, default=1, help="开始折（含）")
    parser.add_argument("--fold_end", type=int, default=5, help="结束折（含）")
    return parser.parse_args()

# 后处理函数
def post_process_mask(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask but got shape {mask_np.shape}")
    mask_np = mask_np.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_np)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)
    filled_mask = binary_fill_holes(largest_component).astype(np.uint8)
    return filled_mask

# 数据集
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
        image_dir = os.path.join(self.image_root, uid)
        mask_dir = os.path.join(self.mask_root, uid)

        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        assert len(image_files) == 1, f"ID {uid} 图片文件夹里不是唯一图片"
        image_path = os.path.join(image_dir, image_files[0])

        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        assert len(mask_files) == 1, f"ID {uid} 掩码文件夹里不是唯一掩码"
        mask_path = os.path.join(mask_dir, mask_files[0])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)

# 训练函数
def train(fold, args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_df = pd.read_excel(os.path.join(args.excel_dir, f"fold{fold}_train.xlsx"))
    train_dataset = LungUltrasoundDataset(train_df, args.image_root, args.mask_root, args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = MedNeXt(
        in_channels=args.in_channels,
        n_channels=32,
        n_classes=args.num_classes,
        exp_r=2,
        kernel_size=3,
        deep_supervision=True,
        do_res=False,
        do_res_up_down=True,
        block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
    ).to(device)

    if args.pretrained and os.path.exists(args.pretrained):
        state_dict = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ 加载预训练模型成功：{args.pretrained}")
    else:
        if args.pretrained:
            print(f"⚠️ 未找到预训练模型文件：{args.pretrained}")

    criterion = FocalDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.num_epochs} - Train"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)

            if isinstance(outputs, list):
                loss = 0.0
                for out in outputs:
                    out_fg = out[:, 1:2]
                    out_fg = nn.functional.interpolate(out_fg, size=masks.shape[2:], mode='bilinear', align_corners=False)
                    loss += criterion(out_fg, masks)
                loss /= len(outputs)
                outputs_fg = outputs[-1][:, 1:2]
                outputs_fg = nn.functional.interpolate(outputs_fg, size=masks.shape[2:], mode='bilinear', align_corners=False)
            else:
                outputs_fg = outputs[:, 1:2]
                loss = criterion(outputs_fg, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 评估训练集性能
        model.eval()
        total_loss = 0.0
        ious, dices = [], []
        with torch.no_grad():
            for images, masks in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{args.num_epochs} - Eval"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                if isinstance(outputs, list):
                    loss = 0.0
                    for out in outputs:
                        out_fg = out[:, 1:2]
                        out_fg = nn.functional.interpolate(out_fg, size=masks.shape[2:], mode='bilinear', align_corners=False)
                        loss += criterion(out_fg, masks)
                    loss /= len(outputs)
                    outputs_fg = outputs[-1][:, 1:2]
                    outputs_fg = nn.functional.interpolate(outputs_fg, size=masks.shape[2:], mode='bilinear', align_corners=False)
                else:
                    outputs_fg = outputs[:, 1:2]
                    loss = criterion(outputs_fg, masks)

                total_loss += loss.item()

                preds_batch = (torch.sigmoid(outputs_fg) > 0.5).float().cpu().numpy()
                targets_batch = masks.cpu().numpy()

                for pred_np, target_np in zip(preds_batch, targets_batch):
                    pred_np = pred_np.squeeze()
                    target_np = target_np.squeeze()
                    pred_np = post_process_mask(pred_np)

                    preds_flat = pred_np.flatten()
                    targets_flat = target_np.flatten()

                    ious.append(jaccard_score(targets_flat, preds_flat, zero_division=0))
                    dices.append(f1_score(targets_flat, preds_flat, zero_division=0))

        print(f"Fold {fold} Epoch {epoch+1} | "
              f"Train Loss: {total_loss/len(train_loader):.4f} | "
              f"Train mIoU : {np.mean(ious):.4f} | "
              f"Train Dice : {np.mean(dices):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"mednext_fold{fold}.pth"))
    print(f"✅ Fold {fold} 训练完成，模型已保存。\n")

if __name__ == "__main__":
    args = parse_args()
    for fold in range(args.fold_start, args.fold_end + 1):
        train(fold, args)
