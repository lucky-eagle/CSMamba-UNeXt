import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    """Unweighted BCE-with-logits plus per-image soft Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        probability = torch.sigmoid(input)
        num = target.size(0)
        probability = probability.reshape(num, -1)
        target = target.reshape(num, -1)
        intersection = (probability * target).sum(1)
        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probability.sum(1) + target.sum(1) + self.smooth
        )
        dice_loss = 1.0 - dice.mean()
        return bce + dice_loss


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
