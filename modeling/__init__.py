# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .example_model import ResNet18, ResNet50, DeepLabv3

MODELS = {"ResNet18": ResNet18, "ResNet50": ResNet50, "DeepLabv3": DeepLabv3}


def build_model(cfg):
    model = MODELS[cfg.MODEL.ARCHITECTURE](cfg.MODEL.NUM_CLASSES)
    return model.to(cfg.MODEL.DEVICE)
