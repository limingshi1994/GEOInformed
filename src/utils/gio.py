import rasterio


def load_tiff(filename: str):
    """Small helper to load a geotiff"""
    with rasterio.open(filename) as ds:
        data = ds.read()
    return data
