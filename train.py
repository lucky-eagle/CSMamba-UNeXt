import argparse
import os
import random
import time
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations import (Compose, HorizontalFlip, Normalize, OneOf,
                            RandomRotate90, Resize, VerticalFlip)
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from typing import Tuple

# 导入自定义模块（需确保这些文件在当前目录）
import archs
import losses
from dataset import Dataset  # 自定义Dataset类，需实现图像/掩码加载逻辑
from metrics import iou_score  # 自定义IoU/Dice计算函数
from utils import AverageMeter, str2bool  # 自定义工具类


# ----------------------------- 新增：时间格式化工具函数 -----------------------------
def format_time(seconds):
    """将秒数格式化为 时:分:秒 或 分:秒"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


# ----------------------------- 核心度量函数（新增） -----------------------------
def activate_preds(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        return torch.sigmoid(logits)
    else:
        return torch.softmax(logits, dim=1)


def to_one_hot_like(target: torch.Tensor, like: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return target.float()
    else:
        if target.dim() == 4 and target.shape[1] == num_classes:
            return target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if target.shape[1] == 1:
            tgt = target.squeeze(1).long()
        else:
            tgt = target.argmax(dim=1).long()
        B, _, H, W = like.shape
        one_hot = torch.zeros(B, num_classes, H, W, device=target.device, dtype=like.dtype)
        return one_hot.scatter_(1, tgt.unsqueeze(1), 1.0)


def class_balance_weights(target_oh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W = target_oh.shape
    area = target_oh.sum(dim=(0, 2, 3))
    w = 1.0 / (area.pow(2) + eps)
    if torch.isinf(w).any() or w.sum() == 0:
        w = torch.ones_like(w)
    w = w * (C / (w.sum() + eps))
    return w.detach()


def weighted_dice(prob: torch.Tensor, target_oh: torch.Tensor, weights: torch.Tensor = None,
                  eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W = prob.shape
    if weights is None:
        weights = class_balance_weights(target_oh)
    weights = weights.to(prob.device).float()

    N = H * W
    p = prob.view(B, C, N)
    g = target_oh.view(B, C, N)

    inter = (p * g).sum(dim=2)
    denom = p.sum(dim=2) + g.sum(dim=2)
    dice_c = (2 * inter + eps) / (denom + eps)

    dice_c_mean = dice_c.mean(dim=0)
    wdice = (weights * dice_c_mean).sum() / (weights.sum() + eps)
    return wdice


def e_measure_torch(prob: torch.Tensor, target_oh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = prob
    g = target_oh
    pm = p - p.mean(dim=(2, 3), keepdim=True)
    gm = g - g.mean(dim=(2, 3), keepdim=True)

    K = (2 * pm * gm) / (pm.pow(2) + gm.pow(2) + eps)
    E = ((K + 1.0) ** 2) / 4.0
    return E.mean()


def mae_torch(prob: torch.Tensor, target_oh: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prob - target_oh))


# ----------------------------- Structure-Measure 函数（新增） -----------------------------
def _object_similarity(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    roi_mask = gt.bool()
    batch_size = prob.shape[0]
    obj_score = torch.zeros(batch_size, device=prob.device)

    for i in range(batch_size):
        roi = prob[i][roi_mask[i]]
        if roi.numel() == 0:
            obj_score[i] = 0.0
            continue
        x = roi.mean()
        sigma_x = roi.std()
        obj_score[i] = 2.0 * x / (x ** 2 + 1.0 + sigma_x + eps)
    return obj_score.mean()


def S_object(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    prob_fg = prob.clone()
    prob_fg[~gt.bool()] = 0.0
    o_fg = _object_similarity(prob_fg, gt)

    prob_bg = 1.0 - prob
    prob_bg[gt.bool()] = 0.0
    o_bg = _object_similarity(prob_bg, ~gt.bool())

    u = gt.float().mean()
    return u * o_fg + (1 - u) * o_bg


def _centroid(gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = gt.shape
    gt_float = gt.float()
    cols = torch.arange(1, W + 1, device=gt.device).view(1, 1, 1, W).repeat(B, 1, H, 1)
    rows = torch.arange(1, H + 1, device=gt.device).view(1, 1, H, 1).repeat(B, 1, 1, W)

    total = gt_float.sum(dim=(2, 3))
    total[total == 0] = 1.0

    X = (gt_float * cols).sum(dim=(2, 3)) / total
    Y = (gt_float * rows).sum(dim=(2, 3)) / total

    X = X.round().clamp(1, W).long()
    Y = Y.round().clamp(1, H).long()
    return X, Y


def _divide_map(map_tensor: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    B, _, H, W = map_tensor.shape
    lt = torch.zeros_like(map_tensor)
    rt = torch.zeros_like(map_tensor)
    lb = torch.zeros_like(map_tensor)
    rb = torch.zeros_like(map_tensor)

    for i in range(B):
        x, y = X[i], Y[i]
        lt[i, :, :y, :x] = map_tensor[i, :, :y, :x]
        if x < W:
            rt[i, :, :y, x:] = map_tensor[i, :, :y, x:]
        if y < H:
            lb[i, :, y:, :x] = map_tensor[i, :, y:, :x]
        if x < W and y < H:
            rb[i, :, y:, x:] = map_tensor[i, :, y:, x:]
    return lt, rt, lb, rb


def _region_ssim(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, C, H, W = prob.shape
    N = H * W

    x = prob.mean(dim=(2, 3), keepdim=True)
    y = gt.mean(dim=(2, 3), keepdim=True)

    sigma_x2 = ((prob - x) ** 2).sum(dim=(2, 3)) / (N - 1 + eps)
    sigma_y2 = ((gt - y) ** 2).sum(dim=(2, 3)) / (N - 1 + eps)
    sigma_xy = ((prob - x) * (gt - y)).sum(dim=(2, 3)) / (N - 1 + eps)

    x_flat = x.squeeze()
    y_flat = y.squeeze()
    sigma_x2_flat = sigma_x2.squeeze()
    sigma_y2_flat = sigma_y2.squeeze()
    sigma_xy_flat = sigma_xy.squeeze()

    alpha = 4 * x_flat * y_flat * sigma_xy_flat
    beta = (x_flat ** 2 + y_flat ** 2) * (sigma_x2_flat + sigma_y2_flat)

    ssim_score = torch.where(alpha != 0, alpha / (beta + eps), torch.zeros_like(alpha))
    ssim_score = torch.where((alpha == 0) & (beta == 0), torch.ones_like(alpha), ssim_score)
    return ssim_score.mean()


def S_region(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    B, _, H, W = prob.shape
    total_area = H * W

    X, Y = _centroid(gt)

    gt_lt, gt_rt, gt_lb, gt_rb = _divide_map(gt, X, Y)
    prob_lt, prob_rt, prob_lb, prob_rb = _divide_map(prob, X, Y)

    w1 = (Y * X) / total_area
    w2 = (Y * (W - X)) / total_area
    w3 = ((H - Y) * X) / total_area
    w4 = 1.0 - w1 - w2 - w3

    q1 = _region_ssim(prob_lt, gt_lt)
    q2 = _region_ssim(prob_rt, gt_rt)
    q3 = _region_ssim(prob_lb, gt_lb)
    q4 = _region_ssim(prob_rb, gt_rb)

    w1_mean, w2_mean, w3_mean, w4_mean = w1.mean(), w2.mean(), w3.mean(), w4.mean()
    return w1_mean * q1 + w2_mean * q2 + w3_mean * q3 + w4_mean * q4


def structure_measure(prob: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if prob.shape[1] != 1 or gt.shape[1] != 1:
        raise ValueError("Structure-Measure only supports binary classification (C=1). Set num_classes=1.")

    y = gt.float().mean()
    if y == 0:
        x = prob.mean()
        sm_score = 1.0 - x
    elif y == 1:
        x = prob.mean()
        sm_score = x
    else:
        alpha = 0.5
        sm_score = alpha * S_object(prob, gt) + (1 - alpha) * S_region(prob, gt)

    return torch.clamp(sm_score, 0.0, 1.0)


# -------------------------- 1. 命令行参数解析（支持跨数据集与架构配置） --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Image Segmentation Training (No Data Leakage)')

    # 基础配置：模型名称、训练轮次、批次大小
    parser.add_argument('--name', default=None,
                        help='model name (default: {dataset}_{arch}_{wDS/woDS})')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='total epochs to run (default: 100)')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N',
                        help='mini-batch size (default: 16)')

    # 模型配置：架构、输入参数、深度监督
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext',
                        choices=['UNext', 'CSMamba-UNeXt', 'UNet', 'ResUNet'],  # 限制支持的架构
                        help=f'model architecture: {", ".join(["UNext", "CSMamba-UNeXt", "UNet", "ResUNet"])} (default: UNext)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='enable deep supervision (default: False)')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input image channels (3 for RGB, 1 for gray, default: 3)')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of segmentation classes (default: 1 for binary)')
    parser.add_argument('--input_w', default=256, type=int,
                        help='input image width (default: 256)')
    parser.add_argument('--input_h', default=256, type=int,
                        help='input image height (default: 256)')

    # 损失函数配置
    LOSS_NAMES = losses.__all__ + ['BCEWithLogitsLoss']
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES,
                        help='loss function: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # 数据集配置（跨数据集核心参数）
    parser.add_argument('--train_dataset', default='TrainDataset',
                        help='training dataset name (path: inputs/{train_dataset}, default: TrainDataset)')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension (default: .png)')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension (default: .png)')
    parser.add_argument('--mean', default='0.5,0.5,0.5', type=str,
                        help='normalization mean (split by comma, default: 0.5,0.5,0.5)')
    parser.add_argument('--std', default='0.5,0.5,0.5', type=str,
                        help='normalization std (split by comma, default: 0.5,0.5,0.5)')

    # 优化器与调度器配置
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer: Adam | SGD (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR',
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='SGD nesterov momentum (default: False)')

    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'],
                        help='learning rate scheduler (default: CosineAnnealingLR)')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate for scheduler (default: 1e-5)')
    parser.add_argument('--factor', default=0.1, type=float,
                        help='ReduceLROnPlateau factor (default: 0.1)')
    parser.add_argument('--patience', default=2, type=int,
                        help='ReduceLROnPlateau patience (default: 2)')
    parser.add_argument('--milestones', default='50,80', type=str,
                        help='MultiStepLR milestones (split by comma, default: 50,80)')
    parser.add_argument('--gamma', default=2 / 3, type=float,
                        help='MultiStepLR gamma (default: 2/3)')

    # 早停与数据加载配置
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N',
                        help='early stopping epochs (default: -1, disable)')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=41, type=int,
                        help='random seed for data split (default: 41, ensure reproducibility)')

    config = parser.parse_args()

    # 解析归一化参数（字符串转列表）
    config.mean = [float(x) for x in config.mean.split(',')]
    config.std = [float(x) for x in config.std.split(',')]

    # 自动生成模型名称（若未指定）
    if config.name is None:
        ds_suffix = 'wDS' if config.deep_supervision else 'woDS'
        config.name = f"{config.train_dataset}_{config.arch}_{ds_suffix}"

    return config


# -------------------------- 2. 训练函数（单轮epoch训练逻辑） --------------------------
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()  # 切换训练模式（启用Dropout/BatchNorm更新）
    pbar = tqdm(total=len(train_loader), desc='Training')

    for input, target, _ in train_loader:
        # 数据转移到GPU
        input = input.cuda()
        target = target.cuda()

        # 模型前向传播（支持深度监督的双输出）
        main_pred, aux_pred = model(input)

        # 计算损失（深度监督：主损失 + 0.4×辅助损失）
        if config.deep_supervision:
            loss_main = criterion(main_pred, target)
            loss_aux = criterion(aux_pred, target)
            loss = loss_main + 0.4 * loss_aux
        else:
            loss = criterion(main_pred, target)

        # 计算IoU（仅用主预测计算指标）
        iou, _ = iou_score(main_pred, target)  # 假设iou_score返回(IoU, Dice)

        # 梯度下降
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累加指标（按批次大小加权）
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        # 更新进度条
        pbar.set_postfix(OrderedDict([
            ('loss', f"{avg_meters['loss'].avg:.4f}"),
            ('iou', f"{avg_meters['iou'].avg:.4f}")
        ]))
        pbar.update(1)
    pbar.close()

    # 返回单轮训练指标
    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg)
    ])


# -------------------------- 3. 验证/测试函数（新增所有评价指标） --------------------------
def evaluate(config, data_loader, model, criterion, split_name='Val'):
    # 新增指标的AverageMeter
    avg_meters = {
        'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(),
        'wdice': AverageMeter(), 'sm': AverageMeter(), 'e': AverageMeter(), 'mae': AverageMeter()
    }
    model.eval()  # 切换评估模式（禁用Dropout，固定BatchNorm）

    with torch.no_grad():  # 禁用梯度计算，加速推理并减少内存占用
        # 验证集进度条更突出，测试集保持简洁
        desc = f"📊 {split_name} Evaluating (Key Metrics)" if split_name == 'Val' else f"{split_name} Evaluating"
        pbar = tqdm(total=len(data_loader), desc=desc)

        for input, target, _ in data_loader:
            input = input.cuda()
            target = target.cuda()

            # 模型前向传播
            main_pred, aux_pred = model(input)

            # 计算损失（与训练逻辑一致）
            if config.deep_supervision:
                loss_main = criterion(main_pred, target)
                loss_aux = criterion(aux_pred, target)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = criterion(main_pred, target)

            # 计算基础指标（IoU/Dice）
            iou, dice = iou_score(main_pred, target)

            # 激活预测结果（sigmoid/softmax）
            prob = activate_preds(main_pred, config.num_classes)

            # 转换target为one-hot格式（匹配prob的维度和类型）
            target_oh = to_one_hot_like(target, prob, config.num_classes)

            # 计算新增指标
            wdice = weighted_dice(prob, target_oh)
            e = e_measure_torch(prob, target_oh)
            mae = mae_torch(prob, target_oh)

            # 计算Structure-Measure（仅支持二分类）
            if config.num_classes == 1:
                sm = structure_measure(prob, target_oh)
            else:
                sm = torch.tensor(0.0, device=prob.device)  # 多分类时SM指标置0

            # 累加所有指标
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['wdice'].update(wdice.item(), input.size(0))
            avg_meters['sm'].update(sm.item() if config.num_classes == 1 else 0.0, input.size(0))
            avg_meters['e'].update(e.item(), input.size(0))
            avg_meters['mae'].update(mae.item(), input.size(0))

            # 进度条显示：验证集显示所有指标（重点突出），测试集仅显示核心指标
            if split_name == 'Val':
                pbar.set_postfix(OrderedDict([
                    ('loss', f"{avg_meters['loss'].avg:.4f}"),
                    ('iou', f"{avg_meters['iou'].avg:.4f}"),
                    ('dice', f"{avg_meters['dice'].avg:.4f}"),
                    ('wdice', f"{avg_meters['wdice'].avg:.4f}"),
                    ('sm', f"{avg_meters['sm'].avg:.4f}"),
                    ('e', f"{avg_meters['e'].avg:.4f}"),
                    ('mae', f"{avg_meters['mae'].avg:.4f}")
                ]))
            else:
                pbar.set_postfix(OrderedDict([
                    ('loss', f"{avg_meters['loss'].avg:.4f}"),
                    ('iou', f"{avg_meters['iou'].avg:.4f}"),
                    ('dice', f"{avg_meters['dice'].avg:.4f}")
                ]))
            pbar.update(1)
        pbar.close()

    # 返回所有评估指标
    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('wdice', avg_meters['wdice'].avg),
        ('sm', avg_meters['sm'].avg),
        ('e', avg_meters['e'].avg),
        ('mae', avg_meters['mae'].avg)
    ])


# -------------------------- 4. 主函数（重点突出val结果） --------------------------
def main():
    # 1. 解析参数并初始化配置
    config = parse_args()
    config_dict = vars(config)  # 转为字典，方便保存配置
    random.seed(config.seed)  # 固定随机种子，确保数据划分可复现
    torch.manual_seed(config.seed)
    cudnn.benchmark = True  # 启用GPU卷积优化（输入尺寸固定时加速）

    # 2. 创建模型保存目录并保存配置
    save_dir = f"models/{config.name}"
    os.makedirs(save_dir, exist_ok=True)
    # 保存配置到yaml文件（后续评估时复用）
    with open(f"{save_dir}/config.yml", 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)

    # 打印配置信息（方便核对）
    print('=' * 80)
    print(f"📌 Model Name: {config.name}")
    print('-' * 80)
    for k, v in config_dict.items():
        print(f"{k:20}: {v}")
    print('=' * 80)

    # 3. 定义损失函数（支持自定义损失与PyTorch原生损失）
    if config.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config.loss]().cuda()  # 自定义损失（如BCEDiceLoss）

    # 4. 初始化模型（适配CSMamba-UNeXt与其他通用架构）
    print(f"\n=> Creating model: {config.arch}")
    if config.arch == 'CSMamba-UNeXt':
        # CSMamba-UNeXt特殊参数初始化（从配置读取，无则用默认）
        model = archs.CSMamba-UNeXt(
            num_classes=config.num_classes,
            img_size=config.input_w,  # 假设CSMamba-UNeXt用input_w作为边长（正方形输入）
            embed_dims=config_dict.get('embed_dims', [128, 160, 256]),
            drop_path_rate=config_dict.get('drop_path_rate', 0.1),
            drop_rate=config_dict.get('drop_rate', 0.2)
        ).cuda()
    else:
        # 通用架构初始化（按固定参数顺序传递）
        model = archs.__dict__[config.arch](
            num_classes=config.num_classes,
            input_channels=config.input_channels,
            deep_supervision=config.deep_supervision
        ).cuda()

    # 5. 初始化优化器（仅优化可训练参数）
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.lr, weight_decay=config.weight_decay
        )
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(
            params, lr=config.lr, momentum=config.momentum,
            nesterov=config.nesterov, weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported")

    # 6. 初始化学习率调度器
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.min_lr
        )
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.factor, patience=config.patience,
            verbose=True, min_lr=config.min_lr
        )
    elif config.scheduler == 'MultiStepLR':
        milestones = [int(e) for e in config.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=config.gamma
        )
    elif config.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not supported")

    # -------------------------- 7. 数据加载（修复参数重复问题） --------------------------
    # 7.1 定义数据集路径（从参数读取，支持任意数据集）
    base_dir = os.path.join('inputs', config.train_dataset)
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')

    # 校验数据集路径有效性
    all_img_ids = glob(os.path.join(img_dir, '*' + config.img_ext))  # 改名避免与后续ID混淆
    assert len(all_img_ids) > 0, f"No images found in {img_dir} (check dataset name or img_ext)"
    all_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in all_img_ids]  # 提取全量图像ID

    # 7.2 数据划分（80%训练 + 10%验证 + 10%测试）
    # train_ids, temp_ids = train_test_split(
    #    all_img_ids, test_size=0.2, random_state=config.seed, shuffle=True
    # )
    # val_ids, test_ids = train_test_split(
    #    temp_ids, test_size=0.5, random_state=config.seed, shuffle=True
    # )
    # print(f"\nData Split: Total={len(all_img_ids)} | Train={len(train_ids)} | Val={len(val_ids)} | Test={len(test_ids)}")
    # print(f"⚠️  Training Focus: Val Dataset Results (Test Results Only After Training)")
    # print('=' * 80)
    # 7.2 数据划分（90%训练 + 10%验证）
    train_ids, val_ids = train_test_split(
        all_img_ids, test_size=0.1, random_state=config.seed, shuffle=True
    )

    test_ids = []  # 不使用测试集，但保留变量，避免后续代码报错

    print(
        f"\nData Split: Total={len(all_img_ids)} | Train={len(train_ids)} | Val={len(val_ids)} | Test={len(test_ids)}")
    print(f"⚠️  Training Focus: Val Dataset Results (No test set used)")
    print('=' * 80)

    # 7.3 定义数据变换（不变）
    train_transform = Compose([
        RandomRotate90(),
        OneOf([HorizontalFlip(p=0.5), VerticalFlip(p=0.5)], p=1.0),
        Resize(config.input_h, config.input_w),
        Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0)
    ])
    eval_transform = Compose([
        Resize(config.input_h, config.input_w),
        Normalize(mean=config.mean, std=config.std, max_pixel_value=255.0)
    ])

    # 7.4 创建Dataset与DataLoader（修复common_ds_kwargs，移除img_ids）
    # 通用参数字典：仅包含所有数据集共享的参数（无img_ids）
    common_ds_kwargs = dict(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_ext=config.img_ext,
        mask_ext=config.mask_ext,
        num_classes=config.num_classes
    )

    # 训练集：传入专属的train_ids，无参数重复
    train_dataset = Dataset(
        img_ids=train_ids,
        transform=train_transform,
        **common_ds_kwargs
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True, pin_memory=True
    )

    # 验证集：传入专属的val_ids
    val_dataset = Dataset(
        img_ids=val_ids,
        transform=eval_transform,
        **common_ds_kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False, pin_memory=True
    )

    # 测试集：传入专属的test_ids
    test_dataset = Dataset(
        img_ids=test_ids,
        transform=eval_transform,
        **common_ds_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, drop_last=False, pin_memory=True
    )

    # -------------------------- 8. 训练循环（重点突出val结果） --------------------------
    # 初始化日志与最优指标
    log = OrderedDict([
        ('epoch', []), ('lr', []), ('epoch_time', []),
        ('train_loss', []), ('train_iou', []),
        # Val指标（重点）
        ('val_loss', []), ('val_iou', []), ('val_dice', []),
        ('val_wdice', []), ('val_sm', []), ('val_e', []), ('val_mae', []),
        # Test指标（仅训练后填充）
        ('test_loss', []), ('test_iou', []), ('test_dice', []),
        ('test_wdice', []), ('test_sm', []), ('test_e', []), ('test_mae', [])
    ])
    best_val_iou = 0.0  # 以验证集IoU为最优模型标准
    best_val_metrics = {}  # 保存最优验证集所有指标
    early_stopping_trigger = 0  # 早停计数器
    total_training_time = 0.0  # 总训练时间

    print("\n" + "=" * 80)
    for epoch in range(config.epochs):
        print(f"🚀 Epoch [{epoch + 1:3d}/{config.epochs:3d}]")
        print("-" * 80)

        # 记录当前epoch的开始时间
        epoch_start_time = time.time()

        # 8.1 单轮训练（仅训练集）
        train_log = train(config, train_loader, model, criterion, optimizer)
        # 8.2 单轮验证（仅验证集，重点输出）
        val_log = evaluate(config, val_loader, model, criterion, split_name='Val')

        # 计算当前epoch的耗时
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        # 8.3 更新学习率调度器
        if config.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])  # 基于验证集损失调整

        # 8.4 打印当前轮次指标（重点突出Val结果）
        print("\n📊 Training Results:")
        print(f"  Loss: {train_log['loss']:.4f} | IoU: {train_log['iou']:.4f}")

        print("\n🎯 Val Dataset Results (Key Focus):")
        print(f"  Loss: {val_log['loss']:.4f} | IoU: {val_log['iou']:.4f} | Dice: {val_log['dice']:.4f}")
        print(
            f"  W-Dice: {val_log['wdice']:.4f} | SM: {val_log['sm']:.4f} | E-Measure: {val_log['e']:.4f} | MAE: {val_log['mae']:.4f}")
        print(f"\n⏱️  Epoch Time: {format_time(epoch_time)} | Total Time: {format_time(total_training_time)}")
        print(f"📉 Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 8.5 记录最优验证集指标
        if val_log['iou'] > best_val_iou:
            best_val_iou = val_log['iou']
            best_val_metrics = val_log.copy()
            torch.save(model.state_dict(), f"{save_dir}/model_best.pth")
            print(f"\n✅ New Best Model Saved! (Best Val IoU: {best_val_iou:.4f})")
            early_stopping_trigger = 0  # 重置早停计数器
        else:
            early_stopping_trigger += 1
            print(
                f"\n⚠️  No Improvement. Early Stopping Trigger: {early_stopping_trigger}/{config.early_stopping}" if config.early_stopping != -1 else "")

        # 8.6 记录日志（测试集指标暂填NaN）
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        log['epoch'].append(epoch + 1)  # epoch从1开始计数
        log['lr'].append(current_lr)
        log['epoch_time'].append(format_time(epoch_time))
        log['train_loss'].append(train_log['loss'])
        log['train_iou'].append(train_log['iou'])

        # 重点记录Val所有指标
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_wdice'].append(val_log['wdice'])
        log['val_sm'].append(val_log['sm'])
        log['val_e'].append(val_log['e'])
        log['val_mae'].append(val_log['mae'])

        # 测试集指标暂填NaN
        log['test_loss'].append(float('nan'))
        log['test_iou'].append(float('nan'))
        log['test_dice'].append(float('nan'))
        log['test_wdice'].append(float('nan'))
        log['test_sm'].append(float('nan'))
        log['test_e'].append(float('nan'))
        log['test_mae'].append(float('nan'))

        # 8.7 保存日志到CSV
        pd.DataFrame(log).to_csv(f"{save_dir}/log.csv", index=False)

        # 8.8 早停判断（基于验证集指标）
        if config.early_stopping >= 0 and early_stopping_trigger >= config.early_stopping:
            print(f"\n⚠️  Early stopping triggered (no improvement for {config.early_stopping} epochs)")
            break

        # 清理GPU缓存，避免内存溢出
        torch.cuda.empty_cache()
        print("\n" + "-" * 80)

    # -------------------------- 9. 训练结束后简要输出测试集结果 --------------------------
    print("\n" + "=" * 80)
    print("🏁 Training Completed!")
    print(f"⏱️  Total Training Time: {format_time(total_training_time)}")

    # 打印最优验证集指标（重点突出）
    print("\n🏆 Best Val Dataset Results (Key Metrics):")
    print(
        f"  Loss: {best_val_metrics.get('loss', 0):.4f} | IoU: {best_val_metrics.get('iou', 0):.4f} | Dice: {best_val_metrics.get('dice', 0):.4f}")
    print(
        f"  W-Dice: {best_val_metrics.get('wdice', 0):.4f} | SM: {best_val_metrics.get('sm', 0):.4f} | E-Measure: {best_val_metrics.get('e', 0):.4f} | MAE: {best_val_metrics.get('mae', 0):.4f}")

    # 简要输出测试集结果
    print("\n📋 Test Dataset Results (Brief):")
    model.load_state_dict(torch.load(f"{save_dir}/model_best.pth"))
    test_log = evaluate(config, test_loader, model, criterion, split_name='Test')

    # 更新日志的最后一行测试集指标
    log['test_loss'][-1] = test_log['loss']
    log['test_iou'][-1] = test_log['iou']
    log['test_dice'][-1] = test_log['dice']
    log['test_wdice'][-1] = test_log['wdice']
    log['test_sm'][-1] = test_log['sm']
    log['test_e'][-1] = test_log['e']
    log['test_mae'][-1] = test_log['mae']

    # 保存最终日志
    pd.DataFrame(log).to_csv(f"{save_dir}/log.csv", index=False)

    # 简要打印测试集结果（不抢Val的重点）
    print(f"  Loss: {test_log['loss']:.4f} | IoU: {test_log['iou']:.4f} | Dice: {test_log['dice']:.4f}")
    print(
        f"  W-Dice: {test_log['wdice']:.4f} | SM: {test_log['sm']:.4f} | E-Measure: {test_log['e']:.4f} | MAE: {test_log['mae']:.4f}")

    print(f"\n💾 Results Saved To:")
    print(f"  - Best Model: {save_dir}/model_best.pth")
    print(f"  - Training Log (Val/Test Metrics): {save_dir}/log.csv")
    print('=' * 80)


if __name__ == '__main__':
    main()