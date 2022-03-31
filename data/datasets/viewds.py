"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""
from tqdm.auto import tqdm

import numpy as np
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from deepsport_utilities.ds.instants_dataset import DeepSportDatasetSplitter
from deepsport_utilities.ds.instants_dataset.views_transforms import (
    CleverViewRandomCropperTransform,
)
from deepsport_utilities.court import Court
from deepsport_utilities.utils import color_cycle_rgb
from mlworkflow import TransformedDataset, PickledDataset
import torch
from PIL import Image
import torchvision.transforms as T


class GenerateViewDS:
    """Transformed View Random Cropper Dataset"""

    def __init__(
        self,
        vds_picklefile: str = "hg_viewdataset.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),  # 640x360, 480x270
        num_elements: int = 1000,
        data_folder: str = "./VIEWDS",
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "hg_viewdataset.pickle".
            output_shape (Tuple[int, int], optional): _description_. Defaults to (1920, 1080).
            num_elements (int, optional): _description_. Defaults to 1000.
        """
        absolute_path = os.path.abspath(__file__)
        absolute_path = os.path.join(*absolute_path.split("/")[:-3])

        print(f"generating data in: {absolute_path}")
        vds = PickledDataset(os.path.join("/", absolute_path, vds_picklefile))
        kwargs = {}
        kwargs["regenerate"] = True
        self.vds = TransformedDataset(
            vds,
            [
                CleverViewRandomCropperTransform(
                    output_shape=output_shape, **kwargs
                )
            ],
        )
        self.num_elements = num_elements
        self._generate_vdataset(num_elements, data_folder)

    def _generate_vdataset(self, num_elements, data_folder):
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        tkeys = len(self.vds.keys)
        random_keys = np.random.randint(tkeys, size=num_elements)
        for inum, random_key in enumerate(random_keys):
            fname = os.path.join(data_folder, f"{inum}")
            key = self.vds.keys[random_key]
            item = self.vds.query_item(key)
            not_generated_keys = []
            if item is not None:
                np.savez_compressed(
                    fname, image=item.image, calib=item.calib.P
                )
            else:
                not_generated_keys.append((fname, key))
        if not_generated_keys:
            print(f"not_generated_keys: {not_generated_keys}")
            self._give_it_another_try(not_generated_keys)

    def _give_it_another_try(self, not_generated_keys):
        for fname, key in not_generated_keys:
            item = self.vds.query_item(key)
            if item:
                np.savez_compressed(
                    fname, image=item.image, calib=item.calib.P
                )


class GenerateSViewDS:
    def __init__(
        self,
        vds_picklefile: str = "hg_viewdataset_2.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),
        def_min: int = 60,
        def_max: int = 160,
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "hg_viewdataset.pickle".
            output_shape (Tuple[int, int], optional): _description_. Defaults to (1920, 1080).
            num_elements (int, optional): _description_. Defaults to 1000.
        """
        absolute_path = os.path.abspath(__file__)
        absolute_path = os.path.join(*absolute_path.split("/")[:-3])

        print(f"generating data in: {absolute_path}")
        vds = PickledDataset(os.path.join("/", absolute_path, vds_picklefile))
        self.vds = TransformedDataset(
            vds,
            [
                CleverViewRandomCropperTransform(
                    output_shape=output_shape,
                    def_min=def_min,
                    def_max=def_max,
                    regenerate=True,
                )
            ],
        )
        dataset_splitter = DeepSportDatasetSplitter(
            additional_keys_usage="skip"
        )
        (self.train, self.val, self.test) = dataset_splitter(self.vds)


class VIEWDS(torch.utils.data.Dataset):
    """A VIEW dataset that returns images and calib objects."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        num_elements: int = 1000,
    ) -> None:
        """_summary_

        Args:
            path (_type_): _description_
        """
        if download:
            GenerateViewDS(num_elements=num_elements)
        root = "VIEWDS"
        # total = len(os.listdir(root))
        total = num_elements
        if train:
            self.list_IDs = os.listdir(root)[: int(total * 0.8)]
        else:
            self.list_IDs = os.listdir(root)[int(total * 0.8) : total]
        self.path = root
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        fname = self.list_IDs[index]

        # Load data and get label
        item = np.load(os.path.join(self.path, f"{fname}"))
        img = Image.fromarray(item["image"])
        if self.transform is not None:
            img = self.transform(img)
        y = item["calib"].flatten()

        return img, y


class SVIEWDS(torch.utils.data.Dataset):
    """Segmentation VIEW dataset.
    It returns the segmentation target for the court.
    """

    def __init__(self, vds, transform: Optional[Callable] = None):
        "Initialization"
        self.vds = vds
        self.vds_keys = list(vds.keys)
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.vds_keys)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        key = self.vds_keys[index]
        item = self.vds.dataset.query_item(key)
        # Load data and get label
        img = Image.fromarray(item.image)

        court = Court(item.rule_type)
        h, w, c = item.image.shape
        target = np.zeros((h, w), dtype=np.uint8)
        court.draw_lines(target, item.calib, color=None)
        if self.transform is not None:
            img = self.transform(img)
        label = {"target": target, "calibP": item.calib.P}
        return (
            img,
            torch.as_tensor(target, dtype=torch.long),
        )
