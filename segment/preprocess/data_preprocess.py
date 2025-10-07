import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='最小外接矩形裁剪后统一 resize 到固定尺寸')
    parser.add_argument('--mask_root', type=str, default='肺部超声数据/mask', help='mask 根目录')
    parser.add_argument('--image_root', type=str, default='肺部超声数据/image', help='image 根目录')
    parser.add_argument('--output_mask_root', type=str, default='肺部超声病灶周边裁剪数据/mask', help='输出 mask 根目录')
    parser.add_argument('--output_image_root', type=str, default='肺部超声病灶周边裁剪数据/image', help='输出 image 根目录')
    parser.add_argument('--target_size', type=int, default=256, help='resize 目标尺寸')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for patient_id in tqdm(os.listdir(args.mask_root), desc='处理裁剪'):
        mask_id_dir = os.path.join(args.mask_root, patient_id)
        image_id_dir = os.path.join(args.image_root, patient_id)
        if not os.path.isdir(mask_id_dir) or not os.path.isdir(image_id_dir):
            continue

        for fname in os.listdir(mask_id_dir):
            if not fname.lower().endswith('.png'):
                continue

            mask_path = os.path.join(mask_id_dir, fname)
            image_path = os.path.join(image_id_dir, fname)

            if not os.path.exists(image_path):
                print(f'找不到图像: {image_path}，跳过')
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path)

            if mask is None or image is None:
                print(f'读取失败: {fname}（ID: {patient_id}）')
                continue

            contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f'未找到前景轮廓: {fname}（ID: {patient_id}）')
                continue

            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped_mask = mask[y:y+h, x:x+w]
            cropped_image = image[y:y+h, x:x+w]

            resized_mask = cv2.resize(cropped_mask, (args.target_size, args.target_size), interpolation=cv2.INTER_NEAREST)
            resized_image = cv2.resize(cropped_image, (args.target_size, args.target_size), interpolation=cv2.INTER_LINEAR)

            out_mask_dir = os.path.join(args.output_mask_root, patient_id)
            out_image_dir = os.path.join(args.output_image_root, patient_id)
            ensure_dir(out_mask_dir)
            ensure_dir(out_image_dir)

            Image.fromarray(resized_mask).save(os.path.join(out_mask_dir, fname))
            Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)).save(os.path.join(out_image_dir, fname))

    print('✅ 最小外接矩形裁剪 + resize 到固定尺寸 完成！')


if __name__ == '__main__':
    main()


