import math
import torch
import torch.nn.functional as F


def split_tensor(tensor, tile_size=256, offset=256):
    tiles = []
    h, w = tensor.shape[1], tensor.shape[2]
    for y in range(int((h - tile_size) / offset) + 1):
        for x in range(int((w - tile_size) / offset) + 1):
            tiles.append(
                tensor[
                    :,
                    offset * y : offset * y + tile_size,
                    offset * x : offset * x + tile_size,
                ]
            )
    if tensor.is_cuda:
        base_tensor = torch.zeros(tensor.shape, device=tensor.get_device())
    else:
        base_tensor = torch.zeros(tensor.shape)
    return tiles, base_tensor


def split_into_tiles(img, tile_size=256, offset=256):
    h, w = img.shape[1], img.shape[2]

    h_padded = math.ceil((h - tile_size) / offset) * offset + tile_size
    w_padded = math.ceil((w - tile_size) / offset) * offset + tile_size

    pad_height_top = math.ceil((h_padded - h) / 2)
    pad_height_bottom = math.floor((h_padded - h) / 2)
    pad_width_left = math.ceil((w_padded - w) / 2)
    pad_width_right = math.floor((w_padded - w) / 2)

    padding = (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)
    original_type = img.dtype
    if original_type != torch.float32:
        img = img.to(torch.float32)
        type_changed = True
    else:
        type_changed = False
    img = F.pad(img, padding, "reflect")
    if type_changed:
        img = img.to(original_type)
    tiles, _ = split_tensor(img, tile_size=tile_size, offset=offset)
    return tiles
