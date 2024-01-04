import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassCalibrationError

from utils.ece_kde import get_ece_kde
class ECE_metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric = MulticlassCalibrationError(num_classes=14, n_bins=15, norm="l1")

    def forward(self, input, target, **kwargs):
        bs = input.shape[0]
        class_num = input.shape[1]
        if "mask" in kwargs:
            mask = kwargs["mask"]
        else:
            mask = None
        probabilities = torch.softmax(input, dim=1)
        label = torch.argmax(target, dim=1)
        if mask is not None:
            label = label.unsqueeze(1)
            probabilities_reshaped = probabilities.permute(0, 2, 3, 1).reshape(-1, class_num)
            mask_reshaped = mask.contiguous().view(-1)
            valid_pixels = probabilities_reshaped[mask_reshaped]
            valid_label = label[mask]
            if valid_pixels.numel() == 0:
                metric = 0
                metric_track = 0
            else:
                metric = self.metric(valid_pixels, valid_label)
                metric_track = metric.detach().cpu().numpy()
        else:
            probabilities = probabilities.view(bs, class_num, -1)
            probabilities = probabilities.transpose(1, 2)
            probabilities = probabilities.reshape(-1, class_num)
            label = label.view(-1)
            metric = self.metric(probabilities, label)
            metric_track = metric.detach().cpu().numpy()
        return metric, metric_track

