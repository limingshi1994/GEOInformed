import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'CCELosswCalib', 'CELoss']


class CCELosswCalib(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', lmda=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target):
        ce_loss = super(CustomCrossEntropyLoss, self).forward(input, target)
        reg_loss = torch.mean(torch.abs(input))  # Example regularization term
        total_loss = ce_loss + self.alpha * reg_loss
        return total_loss


class CELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', lmda=0.1):
        super().__init__(reduction='mean')
        self.alpha = 'good'
        # self.base_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input, target):
        # ce_loss = self.base_loss(input, target)
        ce_loss = super().forward(input, target)
        return ce_loss

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
