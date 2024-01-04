import os
import re
import copy
import ast

import openeo
import shapefile
import geopandas as gpd
from shapely.geometry import shape, Polygon
from geocube.api.core import make_geocube

import argparse
import datetime
from utils.generate_subkaarts import generate_subkaarts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", default="val", type=str)
    parser.add_argument("-o", "--out-dir", default='allbands_download', required=False, type=str)
    parser.add_argument("-k", "--kaartbladen", default=list(range(1, 44)), nargs="+", type=str)
    parser.add_argument("-t", "--temporal-extent", default=['2022-01-01','2023-01-01'], nargs="+", type=str)
    parser.add_argument(
        "-gt",
        "--gt-file",
        default='../resources/BVM_labeled.zip',
        required=False,
        type=str,
        help="Path to the ground truth raster",
    )
    parser.add_argument(
        "-kb",
        "--kaartbladen-file",
        default='../resources/Kbl4.shp',
        required=False,
        type=str,
        help="Path to the ground kaartbladen shape file.",
    )
    args = parser.parse_args()
    return args


def check_args(args):
    kaartbladen = args.kaartbladen
    temporal_extent = args.temporal_extent

    valid_kaartbladen = True
    if not len(kaartbladen):
        valid_kaartbladen = False
    for kaartblad in kaartbladen:
        if kaartblad not in [str(item) for item in range(1, 43)]:
            valid_kaartbladen = False
    if not valid_kaartbladen:
        raise ValueError(f"The provided kaartbladen: {kaartbladen} argument is invalid")

    valid_temporal_extent = True
    if len(temporal_extent) != 2:
        valid_temporal_extent = False
    if not valid_temporal_extent:
        raise ValueError(
            f"The provided valid_temporal_extent: {temporal_extent} argument is invalid"
        )
    year1, month1, day1 = [int(item) for item in temporal_extent[0].split("-")]
    date1 = datetime.date(year1, month1, day1)
    year2, month2, day2 = [int(item) for item in temporal_extent[1].split("-")]
    date2 = datetime.date(year2, month2, day2)
    days_diff = (date2 - date1).days
    if days_diff < 1:
        valid_temporal_extent = False
    if not valid_temporal_extent:
        raise ValueError(
            f"The provided valid_temporal_extent: {temporal_extent} argument is invalid"
        )

    return


def rasterize_polygon(polygon, raster_path, geom=None):
    raster = polygon.to_crs(epsg="32631")
    if geom is not None:
        raster = make_geocube(
            vector_data=raster,
            measurements=["label"],
            resolution=(-10, 10),
            fill=0,
            geom=geom,
        )
    else:
        raster = make_geocube(
            vector_data=raster, measurements=["label"], resolution=(-10, 10), fill=0
        )
    return raster.rio.to_raster(raster_path)


def return_kaartblad_shape(shp_location, kaartblad_name):
    shapes = shapefile.Reader(
        shp_location, encoding="latin-1"
    )  # reading shapefile with pyshp library
    shapeRecs = shapes.shapeRecords()
    # shape_record = [item for item in shapes.records() if item[2] == kaartblad_name][0]
    kaartblad_name = kaartblad_name.replace('_','/')
    shape_record = [item for item in shapes.records() if kaartblad_name == item[2]][0]
    shape_id = int(
        re.findall(r"\d+", str(shape_record))[0]
    )  # getting feature(s)'s id of that match
    feature = shapeRecs[shape_id].shape.__geo_interface__
    polygon = shape(feature)
    exterior = polygon.exterior.coords.xy
    exterior = tuple(zip(list(exterior[0]), list(exterior[1])))
    bounds = polygon.bounds
    bounds = (bounds[:2], bounds[2:])
    centroid = polygon.centroid.coords.xy
    centroid = (centroid[0][0], centroid[1][0])
    return polygon, exterior, bounds, centroid


def main():
    args = get_args()
    # check_args(args)

    # kaartbladen = [str(item) for item in range(2, 4)]
    kaartbladen = args.kaartbladen

    split = args.split
    subkaart_selector = {
        "train": 0,
        "val": 1,
        "test": 2,
    }
    subkaart_ind = subkaart_selector[split]
    kaartbladen = generate_subkaarts(kaartbladen)[subkaart_ind]

    out_dir = os.path.join(args.out_dir, split)
    # temporal_extent = ("2022-03-01", "2022-05-01")
    temporal_extent = args.temporal_extent
    # out_dir = "../generated_data"
    # gt_file = "../resources/BVM_labeled.zip"
    gt_file = args.gt_file
    # kaartbladen_file = f"../resources/Kbl.shp"
    kaartbladen_file = args.kaartbladen_file

    strict = False
    buffer = True

    out_dir_gt = f"{out_dir}/data_gt"
    out_dir_sat = f"{out_dir}/data_sat"
    timestamp = "->".join(temporal_extent)

    if not os.path.exists(out_dir_gt):
        os.makedirs(out_dir_gt)

    if not os.path.exists(out_dir_sat):
        os.makedirs(out_dir_sat)

    eoconn = openeo.connect("openeo.vito.be").authenticate_oidc()
    cal_census = gpd.read_file(gt_file)
    cal_census = cal_census.rename(columns={"Label": "label"})
    cal_census.set_crs(epsg="31370", inplace=True)
    cal_census.to_crs(epsg="4326", inplace=True)

    assert os.path.exists(kaartbladen_file), "Input file does not exist."
    patches = []
    patches_boundary = []

    for kaartblad in kaartbladen:
        polygon, _, bounds, _ = return_kaartblad_shape(kaartbladen_file, kaartblad)
        if strict:
            loc_rectangle = polygon
        else:
            frame = {}
            frame["west"] = bounds[0][0]
            frame["east"] = bounds[1][0]
            frame["north"] = bounds[1][1]
            frame["south"] = bounds[0][1]
            loc_rectangle = Polygon(
                [
                    (frame["west"], frame["south"]),
                    (frame["west"], frame["north"]),
                    (frame["east"], frame["north"]),
                    (frame["east"], frame["south"]),
                    (frame["west"], frame["south"]),
                ]
            )
            loc_rectangle_buffered = loc_rectangle.buffer(0.1)
        loc_boundary = gpd.GeoDataFrame(
            [1], geometry=[loc_rectangle], crs=cal_census.crs
        )  # to check original size
        if buffer:
            loc_clipped = gpd.clip(
                cal_census, loc_rectangle_buffered, keep_geom_type=False
            )
        else:
            loc_clipped = gpd.clip(cal_census, loc_rectangle, keep_geom_type=False)
        patches.append(loc_clipped)
        patches_boundary.append(loc_boundary)

    for i, kaartblad in enumerate(kaartbladen):
        test_patch = copy.deepcopy(patches[i])
        if test_patch.empty:
            continue
        test_patch_boundary = copy.deepcopy(patches_boundary[i])
        kaartblad = kaartblad.replace('/', '_')
        if not os.path.isfile(f"{out_dir_gt}/gt_kaartblad_{kaartblad}.tiff"):
            rasterize_polygon(
                test_patch,
                f"{out_dir_gt}/gt_kaartblad_{kaartblad}.tiff",
                geom=test_patch_boundary.to_json(),
            )

        spatial_extent = ast.literal_eval(test_patch_boundary.to_json())

        if temporal_extent is not None:
            cube = eoconn.load_collection(
                "TERRASCOPE_S2_TOC_V2",
                spatial_extent=spatial_extent,
                temporal_extent=temporal_extent,
            )
        else:
            cube = eoconn.load_collection(
                "TERRASCOPE_S2_TOC_V2", spatial_extent=spatial_extent
            )

        cube = cube.save_result(format="GTiff")
        job = cube.create_job(title=f"Satellite_timeseries_{kaartblad}_{timestamp}")
        job.start_and_wait()
        results = job.get_results()
        if not os.path.exists(f"{out_dir_sat}/{kaartblad}"):
            os.makedirs(f"{out_dir_sat}/kaartblad_{kaartblad}")

        for asset in results.get_assets():
            if asset.metadata["type"].startswith("image/tiff"):
                asset.download(
                    f"{out_dir_sat}/kaartblad_{kaartblad}/kaartblad_{kaartblad}_{asset.name}"
                )


if __name__ == "__main__":
    main()
