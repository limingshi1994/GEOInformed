import numpy as np


def linear_normalization(
    data,
    min_percentile=1,
    max_percentile=99,
    min_out=0.0,
    max_out=1.0,
    norm_hi=None,
    norm_lo=None,
):
    if (norm_hi is not None) and (norm_lo is not None):
        c = norm_lo[:, None, None]
        d = norm_hi[:, None, None]
    else:
        norm_lo = np.percentile(data, min_percentile, axis=[1, 2])
        norm_hi = np.percentile(data, max_percentile, axis=[1, 2])
        c = norm_lo[:, None, None]
        d = norm_hi[:, None, None]
    data = (data - c) * (max_out - min_out) / (d - c) + min_out
    data = np.clip(data, 0, 1)
    return data


def satellite_normalization_with_cloud_masking(
    data,
    invalid_mask,
    min_percentile=1,
    max_percentile=99,
    min_out=0.0,
    max_out=1.0,
    mask_value=0.0,
    norm_hi=None,
    norm_lo=None,
):
    if invalid_mask.all():
        return mask_value * np.ones_like(data)
    else:
        valid_mask = ~invalid_mask
        if len(valid_mask.shape) == 2:
            valid_mask = np.expand_dims(valid_mask, axis=0)
        if valid_mask.shape[0] != data.shape[0]:
            valid_mask = valid_mask.repeat(data.shape[0], axis=0)
        data_out = []
        for i, (data_channel, mask_channel) in enumerate(zip(data, valid_mask)):
            if not ((norm_hi is not None) and (norm_lo is not None)):
                c = np.percentile(data_channel[mask_channel], min_percentile)
                d = np.percentile(data_channel[mask_channel], max_percentile)
            else:
                c = norm_lo[i]
                d = norm_hi[i]
            data_channel = (data_channel - c) * (max_out - min_out) / (d - c) + min_out
            data_out.append(data_channel)
        data_out = np.stack(data_out)
        data_out = np.clip(data_out, 0, 1)
        data_out[~valid_mask] = mask_value
        return data_out
