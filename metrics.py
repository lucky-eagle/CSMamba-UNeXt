"""Authoritative metrics used for checkpoint selection and final reporting."""

import math

import torch
import torch.nn.functional as F

from standard_segmentation_metrics import StandardSegmentationMetrics


def per_image_iou_dice_from_logits(logits, target, threshold=0.5, eps=1e-5):
    """Return one IoU and Dice value per image, never a batch-pooled score."""
    probability = torch.sigmoid(logits)
    prediction = probability >= threshold
    truth = target >= 0.5
    intersection = (prediction & truth).flatten(1).sum(1).float()
    union = (prediction | truth).flatten(1).sum(1).float()
    prediction_area = prediction.flatten(1).sum(1).float()
    truth_area = truth.flatten(1).sum(1).float()
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (prediction_area + truth_area + eps)
    return iou, dice


@torch.no_grad()
def boundary_f1_from_probabilities(probability, target, threshold=0.5, tolerance_ratio=0.008):
    """Return one Boundary-F1 value per image.

    Predictions are binarized at 0.5. One-pixel foreground boundaries are
    extracted by 3x3 binary erosion, and boundary pixels match within
    ``ceil(0.008 * image diagonal)`` pixels (three pixels at 256x256).
    """
    prediction = (probability >= threshold).to(torch.float32)
    truth = (target >= 0.5).to(torch.float32)

    def foreground_boundary(binary_mask):
        padded = F.pad(binary_mask, (1, 1, 1, 1), mode="constant", value=0)
        eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
        return (binary_mask - eroded).clamp_(0, 1)

    pred_boundary = foreground_boundary(prediction)
    truth_boundary = foreground_boundary(truth)
    height, width = prediction.shape[-2:]
    tolerance_pixels = max(
        1, int(math.ceil(tolerance_ratio * math.hypot(height, width)))
    )
    kernel_size = 2 * tolerance_pixels + 1
    pred_dilated = F.max_pool2d(
        pred_boundary, kernel_size=kernel_size, stride=1, padding=tolerance_pixels
    )
    truth_dilated = F.max_pool2d(
        truth_boundary, kernel_size=kernel_size, stride=1, padding=tolerance_pixels
    )

    pred_count = pred_boundary.flatten(1).sum(1)
    truth_count = truth_boundary.flatten(1).sum(1)
    matched_pred = (pred_boundary * truth_dilated).flatten(1).sum(1)
    matched_truth = (truth_boundary * pred_dilated).flatten(1).sum(1)
    both_empty = (pred_count == 0) & (truth_count == 0)
    precision = torch.where(
        pred_count > 0,
        matched_pred / pred_count.clamp_min(1),
        both_empty.to(torch.float32),
    )
    recall = torch.where(
        truth_count > 0,
        matched_truth / truth_count.clamp_min(1),
        both_empty.to(torch.float32),
    )
    score = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp_min(1e-12),
        torch.zeros_like(precision),
    )
    return score, tolerance_pixels


def iou_score(output, target):
    """Compatibility helper returning image-wise macro means."""
    iou, dice = per_image_iou_dice_from_logits(output, target)
    return float(iou.mean()), float(dice.mean())


__all__ = [
    "StandardSegmentationMetrics",
    "boundary_f1_from_probabilities",
    "per_image_iou_dice_from_logits",
    "iou_score",
]
