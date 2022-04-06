"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
import random
import dataclasses
import copy
from calib3d.calib import parameters_to_affine_transform


import torch
import torchvision.transforms as T
from mlworkflow import TransformedDataset, PickledDataset
import numpy as np
import os
from PIL import Image


from deepsport_utilities.court import Court
from deepsport_utilities.ds.instants_dataset.views_transforms import (
    CleverViewRandomCropperTransform,
    UndistortTransform,
)
from deepsport_utilities.transforms import IncompatibleCropException
from deepsport_utilities.utils import Subset, SubsetType


class GenerateViewDS:
    """Transformed View Random Cropper Dataset"""

    def __init__(
        self,
        vds_picklefile: str = "dataset/camera_calib_viewdataset.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),  # 640x360, 480x270
        num_elements: int = 1000,
        data_folder: str = "./VIEWDS",
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "camera_calib_viewdataset.pickle".
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


class GenerateSViewDS:
    def __init__(
        self,
        vds_picklefile: str = "dataset/camera_calib_viewdataset.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),
        def_min: int = 60,
        def_max: int = 160,
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "camera_calib_viewdataset.pickle".
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
                UndistortTransform(),
                ApplyRandomTransform(
                    output_shape=output_shape,
                    def_min=def_min,
                    def_max=def_max,
                    regenerate=True,
                ),
            ],
        )
        dataset_splitter = DeepSportDatasetSplitter(
            additional_keys_usage="skip"
        )
        (self.train, self.val, self.test) = dataset_splitter(self.vds)


class SVIEWDS(torch.utils.data.Dataset):
    """Segmentation VIEW dataset.
    It returns the segmentation target for the court.
    """

    def __init__(
        self,
        vds: GenerateSViewDS,
        transform: Optional[Callable] = None,
        return_camera: bool = False,
    ):
        "Initialization"
        self.vds = vds
        self.vds_keys = list(vds.keys)
        self.transform = transform
        self.return_camera = return_camera

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
        h, w, _ = item.image.shape
        target = np.zeros((h, w), dtype=np.uint8)
        court.draw_lines(target, item.calib, color=None)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_camera:
            label = {"target": target, "calib": item.calib.P}
            return (img, label)
        return (
            img,
            torch.as_tensor(target, dtype=torch.long),
        )


@dataclasses.dataclass
class DeepSportDatasetSplitter:  # pylint: disable=too-few-public-methods
    validation_pc: int = 15
    additional_keys_usage: str = None
    folds: str = "ABCDE"
    split = {
        "A": ["KS-FR-CAEN", "KS-FR-LIMOGES", "KS-FR-ROANNE"],
        "B": ["KS-FR-NANTES", "KS-FR-BLOIS", "KS-FR-FOS"],
        "C": ["KS-FR-LEMANS", "KS-FR-MONACO", "KS-FR-STRASBOURG"],
        "D": ["KS-FR-GRAVELINES", "KS-FR-STCHAMOND", "KS-FR-POITIERS"],
        "E": ["KS-FR-NANCY", "KS-FR-BOURGEB", "KS-FR-VICHY"],
    }

    @staticmethod
    def count_keys_per_arena_label(keys):
        """returns a dict of (arena_label: number of keys of that arena)"""
        bins = {}
        for key in keys:
            bins[key.arena_label] = bins.get(key.arena_label, 0) + 1
        return bins

    @staticmethod
    def count_keys_per_game_id(keys):
        """returns a dict of (game_id: number of keys of that game)"""
        bins = {}
        for key in keys:
            bins[key.game_id] = bins.get(key.game_id, 0) + 1
        return bins

    def __call__(self, dataset, fold=0):
        keys = list(dataset.keys.all())
        assert 0 <= fold <= len(self.folds) - 1, "Invalid fold index"

        testing_fold = self.folds[fold]
        testing_keys = [
            k for k in keys if k.arena_label in self.split[testing_fold]
        ]

        remaining_arena_labels = [
            label
            for f in self.folds.replace(testing_fold, "")
            for label in self.split[f]
        ]
        remaining_keys = [
            k for k in keys if k.arena_label in remaining_arena_labels
        ]

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        validation_keys = random.sample(
            remaining_keys, len(remaining_keys) * self.validation_pc // 100
        )
        training_keys = [k for k in remaining_keys if k not in validation_keys]

        additional_keys = [
            k
            for k in keys
            if k not in training_keys + validation_keys + testing_keys
        ]

        if additional_keys:
            if self.additional_keys_usage == "testing":
                testing_keys += additional_keys
            elif self.additional_keys_usage == "training":
                training_keys += additional_keys
            elif self.additional_keys_usage == "validation":
                validation_keys += additional_keys
            elif self.additional_keys_usage in ["none", "skip"]:
                pass
            else:
                raise ValueError(
                    "They are additional arena labels that I don't know what to do with. Please tell me the 'additional_keys_usage' argument"
                )

        # Restore random seed
        random.setstate(random_state)

        return [
            Subset(
                name="training",
                subset_type=SubsetType.TRAIN,
                keys=training_keys,
                dataset=dataset,
            ),
            Subset(
                name="validation",
                subset_type=SubsetType.EVAL,
                keys=validation_keys,
                dataset=dataset,
            ),
            Subset(
                name="testing",
                subset_type=SubsetType.EVAL,
                keys=testing_keys,
                dataset=dataset,
            ),
        ]


class ApplyRandomTransform(CleverViewRandomCropperTransform):
    def __init__(self, *args, trials=100, def_min=60, def_max=160, **kwargs):
        """
        def -  definition in pixels per meters. (i.e. 60px/m)
        """
        super().__init__(*args, **kwargs)
        self.trials = trials

    def _apply_transform_once(self, key, item):
        if item is None:
            return None
        parameters = self._get_current_parameters(key, item)
        if parameters is None:
            return None
        keypoints, actual_size, input_shape = parameters
        try:
            angle, x_slice, y_slice = self.compute(
                input_shape, keypoints, actual_size
            )
            flip = self.do_flip and bool(np.random.randint(0, 2))
        except IncompatibleCropException:
            return None

        A = parameters_to_affine_transform(
            angle, x_slice, y_slice, self.output_shape, flip
        )
        if self.regenerate:
            item = copy.deepcopy(item)
        return self._apply_transformation(item, A)

    def __call__(self, key, item):

        for _ in range(self.trials):
            item = self._apply_transform_once(key, item)
            if not isinstance(item.image, type(None)):
                break
        return item
