import os
import re
import copy
import random

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from utils.gio import load_tiff
from utils.cropping import random_crop
from utils.tiling import split_into_tiles
from utils.normalization import satellite_normalization_with_cloud_masking

class SatteliteEvalTiledDataset(nn.Module):
    def __init__(
            self,
            root_dir,
            kaartbladen,
            years,
            months,
            patch_size=256,
            patch_offset=128,
            norm_hi=None,
            norm_lo=None,
            eval_every=1,
            split="val",
            return_full_kaartbladen=False,
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
        self.kaartbladen = kaartbladen
        self.kaartbladen_names = [f"kaartblad_{item}" for item in kaartbladen]
        self.years = years
        self.months = months

        self.patch_size = patch_size
        self.patch_offset = patch_offset
        self.norm_hi = norm_hi
        self.norm_lo = norm_lo

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)
        self.data_list = []
        self.build_data_list()
        self.eval_every = eval_every
        self.return_full_kaartbladen = return_full_kaartbladen

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
                        sample["cloud_path"] = self.data_dict[kaartblad]["cloud_masks"][
                            year
                        ][month][day]
                        self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]

        gt_path = sample["gt_path"]
        sat_path = sample["sat_path"]
        cloud_path = sample["cloud_path"]

        print(f"Index: {idx}")
        print(f"GT Pat: {gt_path}")

        gt = load_tiff(gt_path)
        sat = load_tiff(sat_path)[:3]
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

        if self.return_full_kaartbladen:
            full_sat = copy.deepcopy(sat)
            full_gt = copy.deepcopy(gt)
            full_valid_mask = copy.deepcopy(valid_mask)
            full_cloud_mask = copy.deepcopy(cloud_mask)
            full_label_mask = copy.deepcopy(label_mask)
        else:
            full_sat = torch.empty(size=(1,))
            full_gt = torch.empty(size=(1,))
            full_valid_mask = torch.empty(size=(1,))
            full_cloud_mask = torch.empty(size=(1,))
            full_label_mask = torch.empty(size=(1,))

        sat = split_into_tiles(
            sat, tile_size=self.patch_size, offset=self.patch_offset
        )
        gt = split_into_tiles(
            gt, tile_size=self.patch_size, offset=self.patch_offset
        )
        valid_mask = split_into_tiles(
            valid_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset
        )
        cloud_mask = split_into_tiles(
            cloud_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset
        )
        label_mask = split_into_tiles(
            label_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset
        )

        sat = torch.stack(sat[::self.eval_every])
        gt = torch.stack(gt[::self.eval_every])
        valid_mask = torch.stack(valid_mask[::self.eval_every])
        cloud_mask = torch.stack(cloud_mask[::self.eval_every])
        label_mask = torch.stack(label_mask[::self.eval_every])

        sample = {
            "gt": gt,
            "sat": sat,
            "valid_mask": valid_mask,
            "cloud_mask": cloud_mask,
            "label_mask": label_mask,
            "gt_full": full_gt,
            "sat_full": full_sat,
            "valid_mask_full": full_valid_mask,
            "cloud_mask_full": full_cloud_mask,
            "label_mask_full": full_label_mask,
        }

        return sample