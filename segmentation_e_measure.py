"""Reference threshold-swept enhanced-alignment measure for segmentation."""

import numpy as np
import torch
from py_sod_metrics import Emeasure


class EMeasureMaxAccumulator:
    """Accumulate E_phi curves and return max_t(mean_i(E_phi_i(t)))."""

    def __init__(self, thresholds=256):
        if thresholds != 256:
            raise ValueError("The reference implementation uses 256 thresholds")
        self.curve_sum = np.zeros(256, dtype=np.float64)
        self.count = 0

    @staticmethod
    def _uint8(array):
        return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)

    def update(self, probability, target):
        if torch.is_tensor(probability):
            probability = probability.detach().float().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().float().cpu().numpy()
        if probability.ndim == 3:
            probability = probability[:, None]
        if target.ndim == 3:
            target = target[:, None]
        if probability.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {probability.shape} vs {target.shape}"
            )
        for prediction, ground_truth in zip(probability, target):
            prediction_u8 = self._uint8(np.squeeze(prediction))
            ground_truth_u8 = self._uint8(np.squeeze(ground_truth))
            metric = Emeasure()
            metric.step(prediction_u8, ground_truth_u8, normalize=True)
            curve = np.asarray(metric.get_results()["em"]["curve"], dtype=np.float64)
            # PySOD/CODToolbox divides by (W*H-1). The formula specified for
            # this experiment divides by W*H, so apply the exact correction.
            pixels = ground_truth_u8.size
            curve *= max(pixels - 1, 0) / max(pixels, 1)
            self.curve_sum += curve
            self.count += 1

    def curve(self):
        if not self.count:
            raise ValueError("No samples were accumulated")
        return self.curve_sum / self.count

    def max(self):
        return float(np.max(self.curve()))
