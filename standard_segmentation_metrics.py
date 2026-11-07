"""Shared accumulator for the six non-boundary segmentation metrics."""

import torch

from segmentation_e_measure import EMeasureMaxAccumulator
from standard_sod_metrics import StandardSODAccumulator


class StandardSegmentationMetrics:
    """Per-image metrics followed by a dataset mean.

    IoU and Dice use a 0.5 threshold. MAE uses the continuous sigmoid map.
    F_beta^omega uses beta^2=0.3, S_alpha uses alpha=0.5, and E_phi_max is
    max_t(mean_i(E_phi_i(t))) over the 256-threshold reference curve.
    """

    def __init__(self):
        self.iou_sum = 0.0
        self.dice_sum = 0.0
        self.mae_sum = 0.0
        self.count = 0
        self.sod = StandardSODAccumulator(beta_squared=0.3, s_alpha=0.5)
        self.e_measure = EMeasureMaxAccumulator()

    def update(self, probability, target):
        if probability.shape != target.shape:
            raise ValueError(f"Prediction/target mismatch: {probability.shape} vs {target.shape}")
        binary = probability >= 0.5
        truth = target >= 0.5
        intersection = (binary & truth).flatten(1).sum(1).float()
        union = (binary | truth).flatten(1).sum(1).float()
        pred_area = binary.flatten(1).sum(1).float()
        truth_area = truth.flatten(1).sum(1).float()
        iou = (intersection + 1e-5) / (union + 1e-5)
        dice = (2.0 * intersection + 1e-5) / (pred_area + truth_area + 1e-5)
        mae = torch.abs(probability - target).flatten(1).mean(1)
        self.iou_sum += float(iou.sum())
        self.dice_sum += float(dice.sum())
        self.mae_sum += float(mae.sum())
        self.count += target.shape[0]
        self.sod.update(probability, target)
        self.e_measure.update(probability, target)

    def result(self):
        if not self.count:
            raise RuntimeError("No samples were evaluated")
        result = {
            "mean_iou": self.iou_sum / self.count,
            "mean_dice": self.dice_sum / self.count,
            "mae": self.mae_sum / self.count,
        }
        result.update(self.sod.results())
        result["e_phi_max"] = self.e_measure.max()
        return result

    # Compatibility with the original PraNet metric meter API.
    def averages(self):
        return self.result()
