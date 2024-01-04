import os
import re
import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from tqdm import tqdm

from utils.cloud_mask import create_cloud_mask
from utils.gio import load_tiff
from utils.generate_subkaarts import generate_subkaarts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", default="val", type=str)
    parser.add_argument("-k", "--kaartbladen", default=list(range(1,44)), nargs="+", type=str) #modify index with 0:train 1:val 2:test
    parser.add_argument("-y", "--years", default=['2022'], nargs="+", type=str)
    parser.add_argument("-m", "--months", default=['01','02','03','04','05','06''07','08','09','10','11','12'], nargs="+", type=str)
    parser.add_argument("-r", "--root-dir", default='./allbands_download', type=str) #or downloads_230703/val/ or downloads_230703/test/
    parser.add_argument(
        "-ck1",
        "--cloud-kernel-1",
        default=9,
        type=int,
        help="Size of the gaussian kernel used to process the non-cloud mask.",
    )
    parser.add_argument(
        "-ck2",
        "--cloud-kernel-2",
        default=81,
        type=int,
        help="Size of the gaussian kernel used to process the cloud mask.",
    )
    args = parser.parse_args()
    return args


def check_args(args):
    kaartbladen = args.kaartbladen
    split = args.split
    subkaart_selector = {
        "train": 0,
        "val": 1,
        "test": 2,
    }
    subkaart_ind = subkaart_selector[split]
    kaartbladen = generate_subkaarts(kaartbladen)[subkaart_ind]

    years = args.years
    months = args.months
    root_dir = args.root_dir
    root_dir = os.path.join(root_dir, split)

    valid_kaartbladen = True
    if not len(kaartbladen):
        valid_kaartbladen = False
    for kaartblad in kaartbladen:
        print(kaartblad)
        if kaartblad not in [str(item) for item in range(1, 43)]:
            valid_kaartbladen = False
    if not valid_kaartbladen:
        raise ValueError(f"The provided kaartbladen: {kaartbladen} argument is invalid")

    valid_years = True
    if not len(years):
        valid_years = False
    for year in years:
        if year not in [str(item) for item in range(2010, 2023)]:
            valid_years = False
    if not valid_years:
        raise ValueError(f"The provided years: {years} argument is invalid")

    valid_months = True
    if not len(months):
        valid_months = False
    for month in months:
        if month not in [f"{item:02}" for item in range(1, 13)]:
            valid_months = False
    if not valid_months:
        raise ValueError(
            f"The provided months: {months} argument is invalid, the months must be strings representing strings in 'xy' format. eg. '03'"
        )

    valid_root_dir = True
    if not os.path.exists(root_dir):
        valid_root_dir = False
    if not valid_root_dir:
        raise ValueError(
            f"The provided root directory: {valid_root_dir} argument is invalid"
        )

    return


class CloudComputer(nn.Module):
    def __init__(
        self, root_dir, split, kaartbladen, years, months, cloud_kernel_1=9, cloud_kernel_2=81
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
                The structure of the root dir should be like:
                    root_dir/
                        data_gt\
                            gt_kaartblad_1.tiff
                            ...
                            gt_kaartblad_43.tiff
                        
                        data_sat\
                            kaartblad_1
                                kaartblad_1_202X-XX-XXZ.tif
                                ...
                                kaartblad_1_202X-XX-XXZ.tif
                            ...
                            kaartblad_43
                                kaartblad_43_202X-XX-XXZ.tif
                                ...
                                kaartblad_43_202X-XX-XXZ.tif
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        root_dir = os.path.join(root_dir, split)
        self.root_dir = root_dir
        self.kaartbladen = kaartbladen
        self.split = split
        subkaart_selector = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        self.subkaart_ind = subkaart_selector[split]
        self.kaartbladen = generate_subkaarts(self.kaartbladen)[self.subkaart_ind]

        self.gt_dir = f"{root_dir}/data_gt"
        self.sat_dir = f"{root_dir}/data_sat"
        #self.kaartbladen = kaartbladen
        self.kaartbladen_names = [f"kaartblad_{item}" for item in self.kaartbladen]
        self.years = years
        self.months = months

        self.cloud_kernel_1 = cloud_kernel_1
        self.cloud_kernel_2 = cloud_kernel_2

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)
        self.data_list = []
        self.build_data_list()

    def build_data_dict(self):
        print("Building the data dictionary...")
        for gt_file in os.listdir(self.gt_dir):
            gt_file_path = os.path.join(self.gt_dir, gt_file)
            kaartblad_name = re.findall(r"(kaartblad_\w+-\w).", gt_file)[0]
            if kaartblad_name in self.kaartbladen_names:
                self.data_dict[kaartblad_name] = {}
                self.data_dict[kaartblad_name]["gt_path"] = gt_file_path
                self.data_dict[kaartblad_name]["satellite_images"] = {}
                for sat_file in os.listdir(os.path.join(self.sat_dir, kaartblad_name)):
                    if sat_file.endswith(".tif"):
                        sat_file_path = os.path.join(
                            self.sat_dir, kaartblad_name, sat_file
                        )
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z", sat_file
                        )[0]
                        if (
                            year
                            not in self.data_dict[kaartblad_name]["satellite_images"]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ] = {}

                        if (
                            month
                            not in self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][year][
                                month
                            ] = {}

                        self.data_dict[kaartblad_name]["satellite_images"][year][month][
                            day
                        ] = sat_file_path

    """
    'kaartblad_11': {
        'gt_path': 'generated_data_with_scene_classification/data_gt/gt_kaartblad_11.tiff',
        'satellite_images': {
            '2022': {
                '04': {
                    '10': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-10Z.tif',
                    '30': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-30Z.tif',
                    '20': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-20Z.tif',
                    '03': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-03Z.tif',
                    '15': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-15Z.tif',
                    '23': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-23Z.tif',
                    '28': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-28Z.tif',
                    '25': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-04-25Z.tif'
                },
                '03': {
                    '14': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-14Z.tif',
                    '19': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-19Z.tif',
                    '24': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-24Z.tif',
                    '04': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-04Z.tif',
                    '09': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-09Z.tif',
                    '26': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-26Z.tif',
                    '06': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-06Z.tif',
                    '11': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-11Z.tif',
                    '31': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-31Z.tif',
                    '21': 'generated_data_with_scene_classification/data_sat/kaartblad_11/kaartblad_11_openEO_2022-03-21Z.tif'
                }
            }
        }
    }
    """

    def build_data_list(self):
        print("Building the data list...")
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                for month in self.data_dict[kaartblad]["satellite_images"][year].keys():
                    for day in self.data_dict[kaartblad]["satellite_images"][year][
                        month
                    ].keys():
                        sample = {}
                        sample["gt_path"] = self.data_dict[kaartblad]["gt_path"]
                        sample["sat_path"] = self.data_dict[kaartblad][
                            "satellite_images"
                        ][year][month][day]
                        self.data_list.append(sample)

    def __len__(self):
        # Doesn't matter since the data is generated on the fly
        return len(self.data_list)

    def filter_by_year(self, years):
        for kaartblad in self.data_dict.keys():
            self.data_dict[kaartblad]["satellite_images"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["satellite_images"].items()
                if year in years
            }

    def filter_by_month(self, months):
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                self.data_dict[kaartblad]["satellite_images"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["satellite_images"][
                        year
                    ].items()
                    if month in months
                }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]

        gt_path = sample["gt_path"]
        sat_path = sample["sat_path"]
        cloud_path = sat_path.split(".")[0] + "_cloud.png"

        try:
            gt = load_tiff(gt_path)
            sat = load_tiff(sat_path)
            
            mask = sat[4]
            mask = np.expand_dims(mask, axis=0)
            sat = sat[:3]
            # Temporal bottleneck when cloud_kernel_2 > 9
            if not os.path.exists(cloud_path):
                cloud_mask = create_cloud_mask(
                    mask, self.cloud_kernel_1, self.cloud_kernel_2
                )
                Image.fromarray(cloud_mask[0].astype(np.uint8) * 255, mode="L").save(
                    cloud_path
                )
            success = True
        except:
            success = False
        return success, sat_path


def main():
    args = get_args()
    # check_args(args)
    split = args.split
    # kaartbladen = [str(item) for item in range(1, 43)]
    kaartbladen = args.kaartbladen
    # kaartbladen.remove("35")
    # kaartbladen.remove("39")
    # years = ["2021", "2022"]
    years = args.years
    # months = [f"{item:02}" for item in range(1, 13)]
    months = args.months
    # root_dir = "../generated_data_with_scene_classification"
    root_dir = args.root_dir
    # cloud_kernel_1 = 9
    cloud_kernel_1 = args.cloud_kernel_1
    # cloud_kernel_2 = 81
    cloud_kernel_2 = args.cloud_kernel_2

    # Bad kaartbladen: 35 (mismatch, missaligned gt and sat), 43 (missaligned gt and sat) - the other small ones are good
    dataset = CloudComputer(
        root_dir,
        split,
        kaartbladen,
        years,
        months,
        cloud_kernel_1=cloud_kernel_1,
        cloud_kernel_2=cloud_kernel_2,
    )
    for success, sat_path in tqdm(dataset):
        if not success:
            print(f"Failure on: {sat_path}")


if __name__ == "__main__":
    main()
