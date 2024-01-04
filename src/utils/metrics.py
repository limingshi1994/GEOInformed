import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target, **kwargs):

    smooth = 1e-5
    bs = output.shape[0]

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    output = torch.softmax(output, dim=1)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = output_ & target_
    if mask is not None:
        intersection = intersection * mask
    intersection = intersection.view(bs, -1).sum(dim=1).detach().cpu().numpy()

    union = output_ | target_
    if mask is not None:
        union = union * mask
    union = union.view(bs, -1).sum(dim=1).detach().cpu().numpy()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5
    bs = output.shape[0]

    output = torch.sigmoid(output).view(-1).data.cpu()
    target = target.view(-1).data.cpu()
    intersection = (output * target).view(bs, -1).sum(dim=1).numpy()
    output = output.numpy()
    target = target.numpy()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def pixel_accuracy(output, target, **kwargs):

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    # mask = None
    bs = output.shape[0]
    output = torch.softmax(output, dim=1)
    predicted = torch.argmax(output, dim=1)
    label = torch.argmax(target, dim=1)
    correct = (predicted == label).float()
    if mask is not None:
        correct = correct * mask.squeeze(1)
        correct = correct.view(bs, -1).sum(dim=1).detach().cpu().numpy()
        valid = mask.view(bs, -1).sum(dim=1).detach().cpu().numpy()
    else:
        correct = correct.view(bs, -1).sum(dim=1).detach().cpu().numpy()
        valid = torch.ones_like(correct).view(bs, -1).sum(dim=1).detach().cpu().numpy()
    return correct, valid

def calculate_ece_np(output, target, num_bins=20, **kwargs):

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    bs = output.shape[0]
    probs = torch.softmax(output, dim=1)
    predicted = torch.argmax(probs, dim=1)
    gt_label = torch.argmax(target, dim=1)

    x,y,z = torch.meshgrid(torch.arange(8),torch.arange(256),torch.arange(256))
    predicted_probabilities = probs[x, predicted, y, z]

    correct = (predicted == gt_label).float()

    if mask is not None:
        valid_correct = correct * mask.squeeze(1)
        valid_pred_probs = predicted_probabilities * mask.squeeze(1)
        valid = mask.view(bs, -1).sum(dim=1).detach()
    else:
        valid_correct = correct
        valid_pred_probs = predicted_probabilities
        valid = torch.ones_like(valid_correct).view(bs, -1).sum(dim=1).detach()

    # Flatten your softmax probabilities and correct labels (0s and 1s) here
    flattened_pred_probs = valid_pred_probs.view(-1)
    flattened_correct = valid_correct.reshape(-1)
    flattened_vald = valid.reshape(-1)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    n = len(flattened_pred_probs)
    flattened_pred_probs_cpu = flattened_pred_probs.detach().cpu().numpy()
    flattened_correct_cpu = flattened_correct.detach().cpu().numpy()
    flattened_vald_cpu = flattened_vald.detach().cpu().numpy()

    flattened_pred_probs_valid_cpu = flattened_pred_probs_cpu[flattened_vald_cpu]
    flattened_correct_valid_cpu = flattened_correct_cpu[flattened_vald_cpu]

    if len(flattened_pred_probs_valid_cpu):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find indices that belong to the current bin
            in_bin = np.logical_and(flattened_pred_probs_valid_cpu > bin_lower, flattened_pred_probs_valid_cpu <= bin_upper)

            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                # true_positive = flattened_correct[in_bin].detach().cpu().numpy()

                accuracy = np.mean(flattened_correct_valid_cpu[in_bin])
                confidence = np.mean(flattened_pred_probs_valid_cpu[in_bin])

                ece += np.abs(accuracy - confidence) * prop_in_bin

    # to calculate the cumulative ece for all batches in a training epoch,
    # we must note down how many valid pixels are taken into account in each training batch,
    # then use them as weight when taking average


    return ece, valid



def calculate_ece(output, target, num_bins=20, **kwargs):

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    bs = output.shape[0]
    probs = torch.softmax(output, dim=1)
    predicted = torch.argmax(probs, dim=1)
    gt_label = torch.argmax(target, dim=1)

    x,y,z = torch.meshgrid(torch.arange(bs),torch.arange(256),torch.arange(256))
    predicted_probabilities = probs[x, predicted, y, z]

    correct = (predicted == gt_label).float()

    if mask is not None:
        valid_correct = correct * mask.squeeze(1)
        valid_pred_probs = predicted_probabilities * mask.squeeze(1)
        valid = mask.view(bs, -1).sum(dim=1).detach()
    else:
        valid_correct = correct
        valid_pred_probs = predicted_probabilities
        valid = torch.ones_like(valid_correct).view(bs, -1).sum(dim=1).detach()

    # Flatten your softmax probabilities and correct labels (0s and 1s) here
    flattened_pred_probs = valid_pred_probs.view(-1)
    flattened_correct = valid_correct.reshape(-1)
    flattened_vald_mask = mask.reshape(-1)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1).to(flattened_pred_probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    n = len(flattened_pred_probs)
    flattened_pred_probs_valid = flattened_pred_probs[flattened_vald_mask]
    flattened_correct_valid=flattened_correct[flattened_vald_mask]

    if len(flattened_pred_probs_valid):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find indices that belong to the current bin
            in_bin = torch.logical_and(flattened_pred_probs_valid > bin_lower, flattened_pred_probs_valid <= bin_upper)

            prop_in_bin = torch.mean(in_bin.float())

            if prop_in_bin > 0:
                # true_positive = flattened_correct[in_bin].detach().cpu().numpy()

                accuracy = torch.mean(flattened_correct_valid[in_bin])
                confidence = torch.mean(flattened_pred_probs_valid[in_bin])
                ece_bin = torch.abs(accuracy - confidence) * prop_in_bin

                ece += ece_bin.detach().cpu().numpy()

    # to calculate the cumulative ece for all batches in a training epoch,
    # we must note down how many valid pixels are taken into account in each training batch,
    # then use them as weight when taking average

    valid = valid.detach().cpu().numpy()
    return ece, valid


