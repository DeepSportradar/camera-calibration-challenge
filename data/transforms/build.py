# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    """_summary_

    Args:
        cfg (_type_): _description_
        is_train (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    if cfg.INPUT.TRANSFORMS == False:
        transform = T.Compose([T.ToTensor(), normalize_transform])
        return transform
    if is_train:
        transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=cfg.INPUT.SIZE_TRAIN,
                    scale=(
                        cfg.INPUT.MIN_SCALE_TRAIN,
                        cfg.INPUT.MAX_SCALE_TRAIN,
                    ),
                ),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.ToTensor(),
                normalize_transform,
                RandomErasing(
                    probability=cfg.INPUT.PROB,
                    mean=cfg.INPUT.PIXEL_MEAN,
                    sh=0.1,
                ),
            ]
        )
    else:
        transform = T.Compose(
            [T.Resize(cfg.INPUT.SIZE_TEST), T.ToTensor(), normalize_transform]
        )

    return transform
