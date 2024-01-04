import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FIGSIZE = (5, 4)


def show_tiff(
    filename: str,
    figsize=DEFAULT_FIGSIZE,
    vmin=None,
    vmax=None,
    rescale_percentile=97,
    add_colorbar=False,
):
    """Small helper to load a geotiff and visualize it"""
    with rasterio.open(filename) as ds:
        data = ds.read()

    fig, ax = plt.subplots(figsize=figsize)

    if len(data.shape) == 3:
        if data.max() > 500:
            p = np.percentile(data, rescale_percentile, axis=[1, 2])
            data = data / p[:, None, None]
            data = np.clip(data, 0, 1)
        data = np.moveaxis(data, 0, 2)

    im = ax.imshow(data, vmin=vmin, vmax=vmax)
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.05)


def main():
    files = os.listdir('/users/psi/mli1/Downloads/Experts_tiles/Experts/WH/1/')
    for f in files:
        show_tiff(f'/users/psi/mli1/Downloads/Experts_tiles/Experts/WH/1/{f}')

if __name__ == '__main__':
    main()