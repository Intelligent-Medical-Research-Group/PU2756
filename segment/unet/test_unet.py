import os
import argparse
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
import torch.nn as nn
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Test UNet for segmentation")
    parser.add_argument("--image_root", type=str, required=True, help="图像根目录")
    parser.add_argument("--mask_root", type=str, required=True, help="掩码根目录")
    parser.add_argument("--excel_dir", type=str, required=True, help="五折Excel目录，包含 fold{n}_test.xlsx")
    parser.add_argument("--weights_dir", type=str, required=True, help="UNet 权重目录，包含 unet_fold{n}.pth")
    parser.add_argument("--image_size", type=int, default=256, help="输入尺寸")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--fold_start", type=int, default=1, help="开始折编号（含）")
    parser.add_argument("--fold_end", type=int, default=5, help="结束折编号（含）")
    parser.add_argument("--vis_samples", type=int, default=10, help="可视化抽样数量（用于保存对比图）")
    parser.add_argument("--output_dir", type=str, default="visual_comparison", help="可视化输出目录")
    return parser.parse_args()

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

# 1通道 UNet 模型定义
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = UNet._block(features * 16, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = UNet._block(features * 8, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet._block(features * 4, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
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

# 加载模型
def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 可视化叠加掩码
def overlay_mask(image_gray, mask, color=(0, 255, 0)):
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()
    overlay[mask == 1] = color
    combined = cv2.addWeighted(image_color, 0.7, overlay, 0.3, 0)
    return combined

# 评估某一折
def evaluate_fold(fold, args, device):
    print(f"\n🔎 正在评估 Fold {fold}...")
    output_compare_dir = os.path.join(args.output_dir, f"fold{fold}")
    ensure_dir(output_compare_dir)

    val_df = pd.read_excel(os.path.join(args.excel_dir, f"fold{fold}_test.xlsx"))
    model = load_model(os.path.join(args.weights_dir, f"unet_fold{fold}.pth"), device)
    vis_df = val_df.sample(n=min(args.vis_samples, len(val_df)), random_state=fold * 100).reset_index(drop=True) if args.vis_samples > 0 else pd.DataFrame()

    all_preds, all_targets = [], []

    # 可视化样本已根据参数在上方抽样到 vis_df

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Fold {fold} Testing"):
        uid = str(row["检查流水号"])

        image_dir = os.path.join(args.image_root, uid)
        mask_dir = os.path.join(args.mask_root, uid)

        image_file = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_file = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_file or not mask_file:
            print(f"⚠️ {uid} 图像或掩码缺失")
            continue

        image_path = os.path.join(image_dir, image_file[0])
        mask_path = os.path.join(mask_dir, mask_file[0])

        image_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image_raw is None or mask_raw is None:
            print(f"⚠️ 无法读取图像: {uid}")
            continue

        image_resized = cv2.resize(image_raw, (args.image_size, args.image_size))
        mask_resized = cv2.resize(mask_raw, (args.image_size, args.image_size))
        target = (mask_resized > 127).astype(np.int64)

        image_tensor = torch.tensor(image_resized / 255.0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)  # [1, 1, H, W]
            output = nn.functional.interpolate(output, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
            pred = (output > 0.5).int().squeeze().cpu().numpy()

        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())

        # 可视化
        if args.vis_samples > 0 and not vis_df.empty and uid in vis_df["检查流水号"].astype(str).values:
            vis_pred = overlay_mask(image_resized, pred, color=(0, 0, 255))     # 红色预测
            vis_target = overlay_mask(image_resized, target, color=(0, 255, 0))  # 绿色真实
            combined = np.hstack([
                cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR),
                vis_pred,
                vis_target
            ])
            save_path = os.path.join(output_compare_dir, f"{uid}_compare.png")
            cv2.imwrite(save_path, combined)

    # 前景评估
    miou = jaccard_score(all_targets, all_preds, labels=[1], average='macro')
    dice = f1_score(all_targets, all_preds, labels=[1], average='macro')

    print(f"\n✅ Fold {fold} 评估完成")
    print(f"mIoU: {miou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"可视化图像保存在：{output_compare_dir}\n")
    return miou, dice

# 多折测试
def test_all_folds(args):
    all_miou, all_dice = [], []
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for fold in range(args.fold_start, args.fold_end + 1):
        miou, dice = evaluate_fold(fold, args, device)
        all_miou.append(miou)
        all_dice.append(dice)

    print("\n📊 所有折平均评估结果：")
    print(f"平均 mIoU : {np.mean(all_miou):.4f} ± {np.std(all_miou):.4f}")
    print(f"平均 Dice : {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")

if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.output_dir)
    test_all_folds(args)
