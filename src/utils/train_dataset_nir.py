import os
import re
import random

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from utils.gio import load_tiff
from utils.cropping import random_pixel_uniform_crop
from utils.normalization import satellite_normalization_with_cloud_masking
from utils.generate_subkaarts import generate_subkaarts


class SatteliteTrainDataset(nn.Module):
    def __init__(
            self,
            root_dir,
            kaartbladen,
            years,
            months,
            patch_size=256,
            norm_hi=None,
            norm_lo=None,
            split="train",
            preload_gt_flag=False,
            preload_sat_flag=False,
            preload_cloud_flag=False,
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

        self.root_dir = root_dir
        self.split = split
        self.gt_dir = f"{root_dir}/{split}/data_gt"
        self.sat_dir = f"{root_dir}/{split}/data_sat"

        subkaart_selector = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        self.subkaart_ind = subkaart_selector[split]
        self.kaartbladen = generate_subkaarts(kaartbladen)[self.subkaart_ind]
        self.kaartbladen_names = [f"kaartblad_{item}" for item in self.kaartbladen]

        self.years = years
        self.months = months

        self.patch_size = patch_size
        self.norm_hi = norm_hi
        self.norm_lo = norm_lo

        self.preload_gt_flag = preload_gt_flag
        self.preload_sat_flag = preload_sat_flag
        self.preload_cloud_flag = preload_cloud_flag

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)
        self.filter_by_empty()

        if preload_gt_flag:
            self.preload_gt()

        if preload_sat_flag:
            self.preload_sat()

        if preload_cloud_flag:
            self.preload_cloud()

    def preload_gt(self):
        for kaartblad in self.data_dict.keys():
            gt = load_tiff(self.data_dict[kaartblad]["gt_path"])
            self.data_dict[kaartblad]["gt_path"] = gt

    def preload_sat(self):
        for kaartblad in self.data_dict.keys():
            satellite_images = self.data_dict[kaartblad]["satellite_images"]
            for year in satellite_images.keys():
                for month in satellite_images[year].keys():
                    for day in satellite_images[year][month].keys():
                        sat = load_tiff(satellite_images[year][month][day])[[0,1,3],:,:]
                        self.data_dict[kaartblad]["satellite_images"][year][month][day] = sat
    def preload_cloud(self):
        for kaartblad in self.data_dict.keys():
            cloud_masks = self.data_dict[kaartblad]["cloud_masks"]
            for year in cloud_masks.keys():
                for month in cloud_masks[year].keys():
                    for day in cloud_masks[year][month].keys():
                        cloud = np.array(Image.open(cloud_masks[year][month][day]))
                        self.data_dict[kaartblad]["cloud_masks"][year][month][day] = cloud
    def build_data_dict(self):
        print("Building the data dictionary...")
        for gt_file in os.listdir(self.gt_dir):
            gt_file_path = os.path.join(self.gt_dir, gt_file)
            kaartblad_name = re.findall(r"(kaartblad_\w+-\w).", gt_file)[0]
            if kaartblad_name in self.kaartbladen_names:
                self.data_dict[kaartblad_name] = {}
                self.data_dict[kaartblad_name]["gt_path"] = gt_file_path
                self.data_dict[kaartblad_name]["satellite_images"] = {}
                self.data_dict[kaartblad_name]["cloud_masks"] = {}
                for file in os.listdir(os.path.join(self.sat_dir, kaartblad_name)):
                    if file.endswith(".tif"):
                        sat_file_path = os.path.join(self.sat_dir, kaartblad_name, file)
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z", file
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
                    elif file.endswith(".png"):
                        cloud_file_path = os.path.join(
                            self.sat_dir, kaartblad_name, file
                        )
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z_cloud", file
                        )[0]
                        if year not in self.data_dict[kaartblad_name]["cloud_masks"]:
                            self.data_dict[kaartblad_name]["cloud_masks"][year] = {}

                        if (
                                month
                                not in self.data_dict[kaartblad_name]["cloud_masks"][year]
                        ):
                            self.data_dict[kaartblad_name]["cloud_masks"][year][
                                month
                            ] = {}
                        self.data_dict[kaartblad_name]["cloud_masks"][year][month][
                            day
                        ] = cloud_file_path

    def __len__(self):
        # Doesn't matter since the data is generated on the fly
        return 10000000

    def filter_by_year(self, years):
        for kaartblad in self.data_dict.keys():
            self.data_dict[kaartblad]["satellite_images"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["satellite_images"].items()
                if year in years
            }
            self.data_dict[kaartblad]["cloud_masks"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["cloud_masks"].items()
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
            for year in self.data_dict[kaartblad]["cloud_masks"].keys():
                self.data_dict[kaartblad]["cloud_masks"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["cloud_masks"][
                        year
                    ].items()
                    if month in months
                }

    def filter_by_empty(self):
        keys_to_delete = []
        for kaartblad in self.data_dict.keys():
            gt_path = self.data_dict[kaartblad]["gt_path"]
            gt = load_tiff(gt_path)
            if gt.max() == 0:
                keys_to_delete.append(kaartblad)
        for key in keys_to_delete:
            del self.data_dict[key]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            kaartbladen = list(self.data_dict.keys())
            kaartblad = random.choice(kaartbladen)
            years = list(self.data_dict[kaartblad]["satellite_images"].keys())
            year = random.choice(years)
            months = list(self.data_dict[kaartblad]["satellite_images"][year].keys())
            month = random.choice(months)
            days = list(
                self.data_dict[kaartblad]["satellite_images"][year][month].keys()
            )
            day = random.choice(days)

            if self.preload_gt_flag:
                gt = self.data_dict[kaartblad]["gt_path"]
            else:
                gt_path = self.data_dict[kaartblad]["gt_path"]
                gt = load_tiff(gt_path)

            if self.preload_sat_flag:
                sat = self.data_dict[kaartblad]["satellite_images"][year][month][day]  # we already took [:3]
            else:
                sat_path = self.data_dict[kaartblad]["satellite_images"][year][month][day]
                sat = load_tiff(sat_path)[[0,1,3],:,:]

            if self.preload_cloud_flag:
                cloud_mask = self.data_dict[kaartblad]["cloud_masks"][year][month][day]
            else:
                cloud_path = self.data_dict[kaartblad]["cloud_masks"][year][month][day]
                cloud_mask = np.array(Image.open(cloud_path))

            gtshp = gt.shape
            satshp = sat.shape
            cloudshp = cloud_mask.shape
            widths = [gtshp[1], satshp[1], cloudshp[0]]
            heights = [gtshp[2], satshp[2], cloudshp[1]]
            w_min = min(widths)
            h_min = min(heights)
            gt = gt[:, :w_min, :h_min]
            sat = sat[:, :w_min, :h_min]
            cloud_mask = cloud_mask[:w_min, :h_min]

            cloud_mask = np.expand_dims(cloud_mask, axis=0)
            cloud_mask = cloud_mask > 0
            nolabel_mask = gt == 0
            try:
                invalid_mask = np.logical_or(cloud_mask, nolabel_mask)
            except:
                invalid_mask = cloud_mask
                print(gt_path, sat_path, cloud_path)

            # Normalize input data using linear normalization and cloud masking
            # Do this before any narrowing down so that we use the largest possible area to compute the histogram
            sat = satellite_normalization_with_cloud_masking(
                sat,
                cloud_mask,
                min_percentile=1,
                max_percentile=99,
                mask_value=1.0,
                norm_hi=self.norm_hi,
                norm_lo=self.norm_lo,
            )

            gt = torch.tensor(gt, dtype=torch.long)
            sat = torch.tensor(sat, dtype=torch.float32)
            valid_mask = torch.tensor(invalid_mask, dtype=torch.bool).logical_not()
            cloud_mask = torch.tensor(cloud_mask, dtype=torch.bool)
            label_mask = torch.tensor(nolabel_mask, dtype=torch.bool).logical_not()
            # Get a crop
            sat, gt, valid_mask, cloud_mask, label_mask = random_pixel_uniform_crop(
                sat,
                gt,
                valid_mask,
                cloud_mask,
                label_mask,
                self.patch_size,
                self.patch_size
            )
            valid_ratio = (valid_mask.sum() / valid_mask.numel()).item()
            # If there is at least one pixel that is labeled and not clouded we break and fetch the sample
            if valid_ratio > 0:
                break

        sample = {
            "gt": gt,
            "sat": sat,
            "valid_mask": valid_mask,
            "cloud_mask": cloud_mask,
            "label_mask": label_mask,
        }
        return sample