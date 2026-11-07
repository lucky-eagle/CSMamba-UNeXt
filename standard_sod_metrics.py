"""Standard soft foreground-map metrics used by polyp segmentation papers.

The implementation delegates to PySODMetrics, which is verified against
Deng-Ping Fan's MATLAB CODToolbox. Predictions are converted to uint8 and
evaluated image-by-image, matching the reference evaluation protocol.
"""

import numpy as np
import torch
from py_sod_metrics import Smeasure, WeightedFmeasure


class StandardSODAccumulator:
    """Accumulate standard weighted F-measure and S-measure.

    ``beta_squared`` is the coefficient denoted by beta^2 in the paper
    formula. Polyp-segmentation papers conventionally use beta^2=0.3.
    """

    def __init__(self, beta_squared=0.3, s_alpha=0.5):
        self.beta_squared = float(beta_squared)
        self.s_alpha = float(s_alpha)
        self.weighted_f = WeightedFmeasure(beta=self.beta_squared)
        self.s_measure = Smeasure(alpha=self.s_alpha)
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
        for pred, gt in zip(probability, target):
            pred_u8 = self._uint8(np.squeeze(pred))
            gt_u8 = self._uint8(np.squeeze(gt))
            self.weighted_f.step(pred_u8, gt_u8, normalize=True)
            self.s_measure.step(pred_u8, gt_u8, normalize=True)
            self.count += 1

    def results(self):
        if not self.count:
            raise RuntimeError("No samples were supplied to StandardSODAccumulator")
        return {
            "f_beta_omega": float(self.weighted_f.get_results()["wfm"]),
            "s_alpha": float(self.s_measure.get_results()["sm"]),
        }
