import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    #a = np.array([1,2,3])
    #b = np.array([1,3,2])
    #print((a == b).sum())
    #输出结果为1


    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)
    return iou, dice


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

# 加权Dice计算
def weighted_dice(pred, target, weight=None, num_classes=3):
    """
    计算加权Dice系数，针对多分类任务
    pred: 预测结果，shape为 (B, C, H, W)
    target: 真实标签，shape为 (B, C, H, W)
    weight: 每个类别的权重，shape为 (C,)
    num_classes: 类别数
    """
    smooth = 1e-6  # 防止除零错误
    dice_score = 0.0
    for c in range(num_classes):
        p = pred[:, c, :, :]
        g = target[:, c, :, :]

        intersection = torch.sum(p * g)
        union = torch.sum(p) + torch.sum(g)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        if weight is not None:
            dice_score += weight[c] * dice
        else:
            dice_score += dice

    return dice_score / num_classes


# 结构相似度（SSIM）计算
def ssim_index(pred, target):
    """
    计算结构相似度（SSIM）
    pred: 预测结果，shape为 (B, C, H, W)
    target: 真实标签，shape为 (B, C, H, W)
    """
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 (B, H, W, C) 格式
    target = target.cpu().numpy().transpose(0, 2, 3, 1)

    ssim_value = 0.0
    for b in range(pred.shape[0]):
        for c in range(pred.shape[-1]):
            ssim_value += ssim(pred[b, :, :, c], target[b, :, :, c],
                               data_range=pred[b, :, :, c].max() - pred[b, :, :, c].min())

    return ssim_value / (pred.shape[0] * pred.shape[-1])  # 返回平均SSIM值


# 增强对齐指标（E-measure）计算
def e_measure(pred, target):
    """
    计算增强对齐指标（E-measure）
    pred: 预测结果，shape为 (B, C, H, W)
    target: 真实标签，shape为 (B, C, H, W)
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    B, C, H, W = pred.shape
    e_measure_score = 0.0

    for b in range(B):
        for c in range(C):
            p = pred[b, c, :, :]
            g = target[b, c, :, :]

            # 计算FM（联合偏差）
            FM = (p - np.mean(p)) * (g - np.mean(g))
            phi_FM = 2 * np.mean(FM) / (np.mean(FM) ** 2 + 1)
            e_measure_score += phi_FM

    return e_measure_score / (B * C)  # 返回平均E-measure值


# 示例：在训练过程中计算这些指标
def compute_metrics(pred, target, weight=None, num_classes=3):
    """
    计算加权Dice、SSIM 和 E-measure
    """
    # 计算加权Dice
    weighted_dice_score = weighted_dice(pred, target, weight, num_classes)

    # 计算SSIM
    ssim_score = ssim_index(pred, target)

    # 计算E-measure
    e_measure_score = e_measure(pred, target)

    return weighted_dice_score, ssim_score, e_measure_score