import argparse
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2

from albumentations import Compose, Resize, Normalize

import archs  # 与训练阶段相同的模型定义

# --------------------- 解析推理用参数 ---------------------
def parse_infer_args():
    parser = argparse.ArgumentParser(description="Image Segmentation Inference")

    parser.add_argument('--model_dir', type=str, required=True,
                        help='训练好模型所在目录，例如: models/TrainDataset_UNext_woDS')
    parser.add_argument('--img_path', type=str, required=True,
                        help='待推理的彩色图片路径')
    parser.add_argument('--out_path', type=str, default=None,
                        help='输出mask保存路径（默认与输入同名加后缀 _mask.png）')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='二值化阈值(仅二分类使用，默认0.5)')

    args = parser.parse_args()
    return args

# --------------------- 构建与训练一致的模型 ---------------------
def build_model_from_config(config_dict):
    """
    根据训练时保存的config.yml重建模型结构，并加载最优权重。
    """
    arch = config_dict['arch']
    num_classes = config_dict.get('num_classes', 1)
    input_channels = config_dict.get('input_channels', 3)
    deep_supervision = config_dict.get('deep_supervision', False)

    # 与 train 中 main() 的逻辑保持一致
    if arch == 'CSMamba-UNeXt':
        # 训练时代码中的默认参数也在推理时保持一致
        embed_dims = config_dict.get('embed_dims', [128, 160, 256])
        drop_path_rate = config_dict.get('drop_path_rate', 0.1)
        drop_rate = config_dict.get('drop_rate', 0.2)
        input_w = config_dict.get('input_w', 256)

        model = archs.CSMamba-UNeXt(
            num_classes=num_classes,
            img_size=input_w,
            embed_dims=embed_dims,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate
        )
    else:
        model = archs.__dict__[arch](
            num_classes=num_classes,
            input_channels=input_channels,
            deep_supervision=deep_supervision
        )

    return model

# --------------------- 图片预处理（与训练一致） ---------------------
def get_preprocess_transform(config_dict):
    """
    使用与训练/验证阶段 eval_transform 相同的预处理：
    Resize + Normalize(mean, std, max_pixel_value=255)
    """
    input_h = config_dict.get('input_h', 256)
    input_w = config_dict.get('input_w', 256)
    mean = config_dict.get('mean', [0.5, 0.5, 0.5])
    std = config_dict.get('std', [0.5, 0.5, 0.5])

    transform = Compose([
        Resize(input_h, input_w),
        Normalize(mean=mean, std=std, max_pixel_value=255.0)
    ])
    return transform

# --------------------- 单张图片推理函数 ---------------------
def infer_single_image(model, config_dict, img_path, out_path=None,
                       threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # ---------- 1. 读图：与 Dataset 完全一致 ----------
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Dataset 里就是这样做的
    ori_h, ori_w = img.shape[:2]

    # ---------- 2. 预处理：与训练/验证阶段完全一致 ----------
    # Dataset: augmented = transform(image=img, mask=mask)
    #          img = augmented['image']
    #          img = img.astype('float32') / 255
    transform = get_preprocess_transform(config_dict)
    augmented = transform(image=img)
    img_np = augmented['image']          # H x W x C, 已经 Normalize 过

    # ★ 关键：再除以 255，和 Dataset 保持一致 ★
    img_np = img_np.astype('float32') / 255.0   # 这一行是之前缺失的

    # HWC -> CHW
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # ---------- 3. 前向推理 ----------
    with torch.no_grad():
        outputs = model(img_tensor)

        # 训练中 model(input) 返回 (main_pred, aux_pred)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            main_pred, _ = outputs
        else:
            main_pred = outputs

        num_classes = config_dict.get('num_classes', 1)

        if num_classes == 1:
            prob = torch.sigmoid(main_pred)                    # [B, 1, H, W]
            prob_np = prob.squeeze().cpu().numpy().astype('float32')  # H x W

            # 打印概率分布，方便你对比修正前后的变化
            print(f"prob: min={prob_np.min():.4f}, max={prob_np.max():.4f}, mean={prob_np.mean():.4f}")

            # 背景=黑(0)，分割特征=白(255)
            mask = (prob_np > threshold).astype(np.uint8) * 255

        else:
            prob = torch.softmax(main_pred, dim=1)             # [B, C, H, W]
            pred = prob.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)  # H x W
            fg = (pred > 0).astype(np.uint8)
            mask = fg * 255

    # ---------- 4. resize 回原图大小 ----------
    mask_resized = cv2.resize(mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

    # ---------- 5. 保存 mask ----------
    if out_path is None:
        base, ext = os.path.splitext(img_path)
        out_path = base + "_mask.png"

    cv2.imwrite(out_path, mask_resized)
    print(f"✅ 推理完成，mask 已保存到: {out_path}")



# --------------------- 主入口 ---------------------
def main():
    args = parse_infer_args()

    # 1. 读取 config.yml
    config_path = os.path.join(args.model_dir, "config.yml")
    model_path = os.path.join(args.model_dir, "model.pth")

    assert os.path.exists(config_path), f"找不到配置文件: {config_path}"
    assert os.path.exists(model_path), f"找不到模型权重文件: {model_path}"

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 2. 构建模型并加载权重
    print(f"➡️  使用配置构建模型: arch={config_dict['arch']}, num_classes={config_dict.get('num_classes', 1)}")
    model = build_model_from_config(config_dict)

    # 加载权重（兼容CPU/GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)

    # 兼容 DataParallel 保存的情况（state_dict 里可能有 "module." 前缀）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)

    print(f"✅ 模型权重加载完成: {model_path}")

    # 3. 运行单张图片推理
    infer_single_image(
        model=model,
        config_dict=config_dict,
        img_path=args.img_path,
        out_path=args.out_path,
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
