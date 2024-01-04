import numpy as np
import scipy.signal


def makekernel(size: int) -> np.ndarray:
    assert size % 2 == 1
    kernel_vect = scipy.signal.windows.gaussian(size, std=size / 6.0, sym=True)
    kernel = np.outer(kernel_vect, kernel_vect)
    kernel = kernel / kernel.sum()
    return kernel


def create_cloud_mask(classification, kernel_1=9, kernel_2=81):
    if len(classification.shape) == 3:
        expand_back = True
        classification = classification[0]
    else:
        expand_back = False
    # Keep useful pixels (4=vegetation, 5=not vegetated, 6=water, 7=cloud low probability)
    cloud_mask_1 = ~(
        (classification == 4)
        | (classification == 5)
        | (classification == 6)
        | (classification == 7)
    )
    cloud_mask_1 = cloud_mask_1.astype(float)
    cloud_mask_1 = scipy.ndimage.convolve(cloud_mask_1, makekernel(kernel_1))
    cloud_mask_1 = cloud_mask_1 > 0.057
    #     cloud_mask_1 = cloud_mask_1.astype(float)

    # Remove cloud pixels (3=cloud shadows, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus)
    cloud_mask_2 = (
        (classification == 3)
        | (classification == 8)
        | (classification == 9)
        | (classification == 10)
    )
    cloud_mask_2 = cloud_mask_2.astype(float)
    cloud_mask_2 = scipy.ndimage.convolve(cloud_mask_2, makekernel(kernel_2))
    cloud_mask_2 = cloud_mask_2 > 0.1
    #     cloud_mask_2 = cloud_mask_2.astype(float)

    cloud_mask = cloud_mask_1 | cloud_mask_2
    #     cloud_mask = cloud_mask.astype(float)
    if expand_back:
        cloud_mask = np.expand_dims(cloud_mask, axis=0)

    return cloud_mask
