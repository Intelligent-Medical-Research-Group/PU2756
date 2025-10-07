import os
import argparse
import pandas as pd
import random

# 设置随机种子，保证可复现
random.seed(42)

# 存储所有样本路径
data_pairs = []

def parse_args():
    parser = argparse.ArgumentParser(description='从公开数据集生成预训练对 (image, mask) 的Excel')
    parser.add_argument('--busi_root', type=str, default='Dataset_BUSI/Dataset_BUSI_with_GT', help='BUSI 数据集根目录')
    parser.add_argument('--ddti_img_dir', type=str, default='DDTI/1_or_data/image', help='DDTI 图像目录')
    parser.add_argument('--ddti_mask_dir', type=str, default='DDTI/1_or_data/mask', help='DDTI 掩码目录')
    parser.add_argument('--thyroid_img_dir', type=str, default='Thyroid Dataset/tg3k/thyroid-image', help='甲状腺图像目录')
    parser.add_argument('--thyroid_mask_dir', type=str, default='Thyroid Dataset/tg3k/thyroid-mask', help='甲状腺掩码目录')
    parser.add_argument('--train_out', type=str, default='train.xlsx', help='训练Excel输出路径')
    parser.add_argument('--test_out', type=str, default='test.xlsx', help='测试Excel输出路径')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    return parser.parse_args()

def collect_busi(busi_root, pairs):
    if not os.path.isdir(busi_root):
        return
    for category in os.listdir(busi_root):
        category_path = os.path.join(busi_root, category)
        if not os.path.isdir(category_path):
            continue
        for file in os.listdir(category_path):
            if "_mask" not in file and file.endswith('.png'):
                image_path = os.path.join(category_path, file)
                mask_path = os.path.join(category_path, file.replace('.png', '_mask.png'))
                if os.path.exists(mask_path):
                    pairs.append((image_path, mask_path))

def collect_ddti(ddti_img_dir, ddti_mask_dir, pairs):
    if not os.path.isdir(ddti_img_dir) or not os.path.isdir(ddti_mask_dir):
        return
    for file in os.listdir(ddti_img_dir):
        if file.lower().endswith('.png'):
            image_path = os.path.join(ddti_img_dir, file)
            mask_path = os.path.join(ddti_mask_dir, file)
            if os.path.exists(mask_path):
                pairs.append((image_path, mask_path))

def collect_thyroid(thyroid_img_dir, thyroid_mask_dir, pairs):
    if not os.path.isdir(thyroid_img_dir) or not os.path.isdir(thyroid_mask_dir):
        return
    for file in os.listdir(thyroid_img_dir):
        if file.endswith('.jpg'):
            image_path = os.path.join(thyroid_img_dir, file)
            mask_path = os.path.join(thyroid_mask_dir, file)
            if os.path.exists(mask_path):
                pairs.append((image_path, mask_path))

def main():
    args = parse_args()
    random.seed(args.seed)

    pairs = []
    collect_busi(args.busi_root, pairs)
    collect_ddti(args.ddti_img_dir, args.ddti_mask_dir, pairs)
    collect_thyroid(args.thyroid_img_dir, args.thyroid_mask_dir, pairs)

    random.shuffle(pairs)
    split_idx = int(args.train_ratio * len(pairs))
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]

    train_df = pd.DataFrame(train_pairs, columns=['image_path', 'mask_path'])
    test_df = pd.DataFrame(test_pairs, columns=['image_path', 'mask_path'])

    train_df.to_excel(args.train_out, index=False)
    test_df.to_excel(args.test_out, index=False)

    print(f'样本总数: {len(pairs)}  训练: {len(train_pairs)}  测试: {len(test_pairs)}')
    print(f'已输出: {args.train_out}, {args.test_out}')

if __name__ == '__main__':
    main()
