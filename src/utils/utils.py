import torch
import numpy as np

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def find_geq_trim(data, threshold=0):
    if len(data.shape) == 3:
        data = data[0]
    nonzeros = np.argwhere(data > threshold)
    (ystart, xstart), (ystop, xstop) = nonzeros.min(0), nonzeros.max(0) + 1
    return (ystart, ystop), (xstart, xstop)


def find_leq_trim(data, threshold=1):
    if len(data.shape) == 3:
        data = data[0]
    nonzeros = np.argwhere(data < threshold)
    (ystart, xstart), (ystop, xstop) = nonzeros.min(0), nonzeros.max(0) + 1
    return (ystart, ystop), (xstart, xstop)


def find_percentiles(data, invalid_mask, min_percentile=1, max_percentile=99):
    normalization = {}
    if invalid_mask.all():
        normalization = {
            "c": [-1] * data.shape[0],
            "d": [-1] * data.shape[0],
            "valid": False,
        }
    elif invalid_mask.any():
        valid_mask = ~invalid_mask
        if len(valid_mask.shape) == 2:
            valid_mask = np.expand_dims(valid_mask, axis=0)
        if valid_mask.shape[0] != data.shape[0]:
            valid_mask = valid_mask.repeat(data.shape[0], axis=0)
        normalization = {"c": [], "d": [], "valid": True}
        for i, (data_channel, mask_channel) in enumerate(zip(data, valid_mask)):
            c = np.percentile(data_channel[mask_channel], min_percentile)
            d = np.percentile(data_channel[mask_channel], max_percentile)
            normalization["c"].append(c)
            normalization["d"].append(d)
        return normalization
    else:
        normalization = {"c": [], "d": [], "valid": True}
        for i, data_channel in enumerate(data):
            c = np.percentile(data_channel, min_percentile)
            d = np.percentile(data_channel, max_percentile)
            normalization["c"].append(c)
            normalization["d"].append(d)
    return normalization


def find_valid_pixels(data, invalid_mask):
    valid_mask = ~invalid_mask
    if len(valid_mask.shape) == 2:
        data = np.expand_dims(data, axis=0)
    if valid_mask.shape[0] != data.shape[0]:
        valid_mask = valid_mask.repeat(data.shape[0], axis=0)
    values = []
    for i, (data_channel, mask_channel) in enumerate(zip(data, valid_mask)):
        values.append(data_channel[mask_channel].ravel())
    values = np.array(values)
    return values
