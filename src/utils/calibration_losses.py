import torch
import torch.nn as nn

from utils.ece_kde import get_ece_kde

__all__ = ['Canonical', 'Marginal', "TopLabel"]
class Canonical(nn.Module):
    def __init__(self, bandwidth, p, downsampling_factor=1):
        super().__init__()
        self.bandwidth = bandwidth
        self.p = p
        self.downsampling_factor = downsampling_factor

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
            probabilities = probabilities[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
            label = label.unsqueeze(1)
            label = label[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
            probabilities_reshaped = probabilities.permute(0, 2, 3, 1).reshape(-1, class_num)
            mask = mask[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
            mask_reshaped = mask.contiguous().view(-1)
            valid_pixels = probabilities_reshaped[mask_reshaped]
            valid_label = label[mask]
            if valid_pixels.numel() == 0:
                loss = 0
                loss_track = 0
            else:
                loss = get_ece_kde(valid_pixels,valid_label,self.bandwidth,self.p,"canonical",input.device)
                loss_track = loss.detach().cpu().numpy()
        else:
            probabilities = probabilities[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
            label = label[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
            probabilities = probabilities.view(bs, class_num, -1)
            probabilities = probabilities.transpose(1, 2)
            probabilities = probabilities.reshape(-1, class_num)
            label = label.view(-1)
            loss = get_ece_kde(probabilities, label, self.bandwidth, self.p,"canonical", input.device)
            loss_track = loss.detach().cpu().numpy()
        return loss, loss_track

class Marginal(nn.Module):
    def __init__(self, bandwidth, p, downsampling_factor=1):
        super().__init__()
        self.bandwidth = bandwidth
        self.p = p
        self.downsampling_factor = downsampling_factor

    def forward(self, input, target, **kwargs):
        bs = input.shape[0]
        if "mask" in kwargs:
            mask = kwargs["mask"]
        else:
            mask = None
        input = input[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        target = target[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        input = input.view(bs, -1)
        target = target.view(bs, -1)
        loss = get_ece_kde(input, target, self.bandwidth, self.p,"marginal", input.device)
        return loss

class TopLabel(nn.Module):
    def __init__(self, bandwidth, p, downsampling_factor=1):
        super().__init__()
        self.bandwidth = bandwidth
        self.p = p
        self.downsampling_factor = downsampling_factor

    def forward(self, input, target, **kwargs):
        bs = input.shape[0]
        if "mask" in kwargs:
            mask = kwargs["mask"]
        else:
            mask = None
        input = input[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        target = target[:, :, ::self.downsampling_factor, ::self.downsampling_factor]
        input = input.view(bs, -1)
        target = target.view(bs, -1)
        loss = get_ece_kde(input, target, self.bandwidth, self.p,"top_label", input.device)
        return loss