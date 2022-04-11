# encoding: utf-8
"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""

import argparse
import os
import sys

import numpy as np

from deepsport_utilities.calib import Calib


sys.path.append(".")
from config import cfg
from data import make_data_loader
from engine.example_evaluation import (
    save_predictions_to_json,
    json_serialisable,
)


def generate_gt(cfg, val_loader):

    print("Start exporting")
    width, height = (
        cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[0],
        cfg.INPUT.MULTIPLICATIVE_FACTOR * cfg.INPUT.GENERATED_VIEW_SIZE[1],
    )
    dumpable_list = [{"width": width, "height": height}]
    for _, ybatch in val_loader:
        calib_gt = Calib.from_P(
            np.squeeze(ybatch["calib"].cpu().numpy().astype(np.float32)),
            width=width,
            height=height,
        )
        data = {}
        for key, value in calib_gt.dict.items():
            data.update({key: json_serialisable(value)})
        dumpable_list.append(data)

    save_predictions_to_json(dumpable_list, "ground_truth_train.json")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Template MNIST Inference"
    )
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = (
        int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    )

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    val_loader = make_data_loader(cfg, is_train=False)

    generate_gt(cfg, val_loader)


if __name__ == "__main__":
    main()
