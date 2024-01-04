import math
import random

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def random_crop(
    img,
    gt,
    valid_mask,
    cloud_mask,
    label_mask,
    height,
    width
):
    assert (
        type(img) == torch.Tensor
        and type(gt) == torch.Tensor
        and type(valid_mask) == torch.Tensor
        and type(cloud_mask) == torch.Tensor
        and type(label_mask) == torch.Tensor
    ), "Must have torch.Tensor as inputs"
    assert (
        len(img.shape) == 3 and len(gt.shape) == 3 and len(valid_mask.shape) and len(cloud_mask.shape) and len(label_mask.shape)
    ), "Must have three dimensional image and masks"
    assert (
        img.shape[1] == gt.shape[1] == valid_mask.shape[1] == cloud_mask.shape[1] == label_mask.shape[1]
    ), "The image and masks must have the same height"
    assert (
        img.shape[2] == gt.shape[2] == valid_mask.shape[2] == cloud_mask.shape[2] == label_mask.shape[2]
    ), "The image and masks must have the same width"

    h, w = img.shape[1], img.shape[2]

    if height > h:
        pad_height_top = math.ceil((height - h) / 2)
        pad_height_bottom = math.floor((height - h) / 2)
    else:
        pad_height_top = 0
        pad_height_bottom = 0

    if width > w:
        pad_width_left = math.ceil((width - w) / 2)
        pad_width_right = math.floor((width - w) / 2)
    else:
        pad_width_left = 0
        pad_width_right = 0

    padding = (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)
    if any(padding):
        img = F.pad(img, padding, "reflect")
        gt = F.pad(gt, padding, "reflect")
        valid_mask = F.pad(valid_mask, padding, "reflect")
        cloud_mask = F.pad(cloud_mask, padding, "reflect")
        label_mask = F.pad(label_mask, padding, "reflect")

    if height > h:
        y = 0
    else:
        y = random.randint(0, h - height)

    if width > w:
        x = 0
    else:
        x = random.randint(0, w - width)

    img = img[:, y : y + height, x : x + width]
    gt = gt[:, y : y + height, x : x + width]
    valid_mask = valid_mask[:, y : y + height, x : x + width]
    cloud_mask = cloud_mask[:, y : y + height, x : x + width]
    label_mask = label_mask[:, y : y + height, x : x + width]
    return img, gt, valid_mask, cloud_mask, label_mask


def random_pixel_uniform_crop(
    img,
    gt,
    valid_mask,
    cloud_mask,
    label_mask,
    height,
    width
):
    assert (
        type(img) == torch.Tensor
        and type(gt) == torch.Tensor
        and type(valid_mask) == torch.Tensor
        and type(cloud_mask) == torch.Tensor
        and type(label_mask) == torch.Tensor
    ), "Must have torch.Tensor as inputs"
    assert (
        len(img.shape) == 3 and len(gt.shape) == 3 and len(valid_mask.shape) and len(cloud_mask.shape) and len(label_mask.shape)
    ), "Must have three dimensional image and masks"
    assert (
        img.shape[1] == gt.shape[1] == valid_mask.shape[1] == cloud_mask.shape[1] == label_mask.shape[1]
    ), "The image and masks must have the same height"
    assert (
        img.shape[2] == gt.shape[2] == valid_mask.shape[2] == cloud_mask.shape[2] == label_mask.shape[2]
    ), "The image and masks must have the same width"

    h, w = img.shape[1], img.shape[2]

    pad_height_top = math.ceil(height - 1)
    pad_height_bottom = math.floor(height - 1)
    pad_width_left = math.ceil(width - 1)
    pad_width_right = math.floor(width - 1)
    padding = (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)

    # Pad img
    img = F.pad(img, padding, "reflect")

    # Pad gt
    # Need to convert gt to float to perform reflect padding
    original_gt_type = gt.dtype
    if original_gt_type != torch.float32:
        gt = gt.to(torch.float32)
        type_changed = True
    else:
        type_changed = False
    gt = F.pad(gt, padding, "reflect")
    if type_changed:
        gt = gt.to(original_gt_type)

    # Pad valid mask
    valid_mask = F.pad(valid_mask, padding, "constant", value=0)

    # Pad cloud mask
    # Need to convert cloud_mask to float to perform reflect padding
    original_cloud_mask_type = cloud_mask.dtype
    if original_cloud_mask_type != torch.float32:
        cloud_mask = cloud_mask.to(torch.float32)
        type_changed = True
    else:
        type_changed = False
    cloud_mask = F.pad(cloud_mask, padding, "reflect")
    if type_changed:
        cloud_mask = cloud_mask.to(original_cloud_mask_type)

    # Pad label mask
    label_mask = F.pad(label_mask, padding, "constant", value=0)

    h_padded, w_padded = img.shape[1], img.shape[2]
    y = random.randint(0, h_padded - height)  # TODO: MAY NEED TO SUBTRACT 1 OR STH - TEST
    x = random.randint(0, w_padded - width)   # TODO: MAY NEED TO SUBTRACT 1 OR STH - TEST

    # Hardcoding for comparison
    # x = 1299
    # y = 707
    # print(f"Random coords: {x}, {y}")


    img = img[:, y : y + height, x : x + width]
    gt = gt[:, y : y + height, x : x + width]
    valid_mask = valid_mask[:, y : y + height, x : x + width]
    cloud_mask = cloud_mask[:, y : y + height, x : x + width]
    label_mask = label_mask[:, y : y + height, x : x + width]
    return img, gt, valid_mask, cloud_mask, label_mask