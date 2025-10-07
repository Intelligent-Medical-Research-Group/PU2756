import os
import argparse
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from mednextv1.MedNextV1 import MedNeXt
import torch.nn as nn
from scipy.ndimage import binary_fill_holes

def parse_args():
    parser = argparse.ArgumentParser(description="Test MedNeXt for segmentation")
    parser.add_argument("--image_root", type=str, required=True, help="图像根目录")
    parser.add_argument("--mask_root", type=str, required=True, help="掩码根目录")
    parser.add_argument("--excel_dir", type=str, required=True, help="五折Excel目录")
    parser.add_argument("--weights_dir", type=str, required=True, help="MedNeXt 权重目录，包含 mednext_fold{n}.pth")
    parser.add_argument("--image_size", type=int, default=256, help="输入尺寸")
    parser.add_argument("--in_channels", type=int, default=1, help="输入通道数")
    parser.add_argument("--num_classes", type=int, default=2, help="类别数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，如 cuda:0 或 cpu")
    parser.add_argument("--fold_start", type=int, default=1, help="开始折（含）")
    parser.add_argument("--fold_end", type=int, default=5, help="结束折（含）")
    parser.add_argument("--vis_samples", type=int, default=10, help="可视化抽样数量")
    parser.add_argument("--output_dir", type=str, default="visual_comparison", help="可视化输出目录")
    return parser.parse_args()

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

# 后处理：保留最大连通域并填充空洞
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

# 加载模型
def load_model(model_path, args, device):
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 可视化叠加函数
def overlay_mask(image_gray, mask, color=(0, 255, 0)):
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = image_color.copy()
    overlay[mask == 1] = color
    combined = cv2.addWeighted(image_color, 0.7, overlay, 0.3, 0)
    return combined

# 单折评估
def evaluate_fold(fold, args, device):
    print(f"\n🔎 正在评估 Fold {fold}...")
    output_compare_dir = os.path.join(args.output_dir, f"fold{fold}")
    ensure_dir(output_compare_dir)

    val_df = pd.read_excel(os.path.join(args.excel_dir, f"fold{fold}_test.xlsx"))
    model = load_model(os.path.join(args.weights_dir, f"mednext_fold{fold}.pth"), args, device)
    vis_df = val_df.sample(n=min(args.vis_samples, len(val_df)), random_state=fold * 100).reset_index(drop=True) if args.vis_samples > 0 else pd.DataFrame()

    all_preds, all_targets = [], []

    # 可视化样本（根据参数抽样，已在上方生成 vis_df）

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Fold {fold} Testing"):
        uid = str(row["检查流水号"])

        image_dir = os.path.join(args.image_root, uid)
        mask_dir = os.path.join(args.mask_root, uid)

        image_file = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        mask_file = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_file or not mask_file:
            print(f"⚠️ 警告: {uid} 图像或掩膜文件缺失")
            continue

        image_path = os.path.join(image_dir, image_file[0])
        mask_path = os.path.join(mask_dir, mask_file[0])

        image_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image_raw is None or mask_raw is None:
            print(f"⚠️ 无法读取图像: {uid}")
            continue

        image_resized = cv2.resize(image_raw, (args.image_size, args.image_size))
        image_tensor = torch.tensor(image_resized, dtype=torch.float32, device=device) / 255.0
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W

        with torch.no_grad():
            outputs = model(image_tensor)
            output = outputs[-1] if isinstance(outputs, list) else outputs
            output = nn.functional.interpolate(output, size=(args.image_size, args.image_size), mode='bilinear', align_corners=False)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # 后处理
        pred = post_process_mask(pred)

        mask_resized = cv2.resize(mask_raw, (args.image_size, args.image_size))
        target = (mask_resized > 127).astype(np.int64)

        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())

        if args.vis_samples > 0 and not vis_df.empty and uid in vis_df["检查流水号"].astype(str).values:
            vis_pred = overlay_mask(image_resized, pred, color=(0, 0, 255))    # 红色预测
            vis_target = overlay_mask(image_resized, target, color=(0, 255, 0))  # 绿色真实
            combined = np.hstack([
                cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR),
                vis_pred,
                vis_target
            ])
            save_path = os.path.join(output_compare_dir, f"{uid}_compare.png")
            cv2.imwrite(save_path, combined)

    # 前景类别评估
    miou = jaccard_score(all_targets, all_preds, labels=[1], average='macro')
    dice = f1_score(all_targets, all_preds, labels=[1], average='macro')

    print(f"✅ Fold {fold} 前景评估")
    print(f"mIoU: {miou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"可视化图保存路径：{output_compare_dir}")

    return miou, dice

# 多折测试（如需可改 range）
def test_all_folds(args):
    all_miou, all_dice = [], []
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    for fold in range(args.fold_start, args.fold_end + 1):
        miou, dice = evaluate_fold(fold, args, device)
        all_miou.append(miou)
        all_dice.append(dice)

    print("\n📊 五折平均评估结果：")
    print(f"mIoU 平均值: {np.mean(all_miou):.4f} ± {np.std(all_miou):.4f}")
    print(f"Dice 平均值: {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.output_dir)
    test_all_folds(args)
