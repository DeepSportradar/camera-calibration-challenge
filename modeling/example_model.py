# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

and

@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""

import numpy as np
import torch.nn.functional as F
import torch
from torch import nn

from layers.conv_layer import conv3x3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.linear = nn.Linear(64 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        backbone = torch.hub.load(
            "pytorch/vision:v0.11.3",
            "resnet50",
            pretrained=True,
        )
        self.features = nn.Sequential(*(list(backbone.children())[:-2]))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.gap(out)
        out = self.flat(out)
        out = self.linear(out)

        return out


def DeepLabv3(outputchannels=1):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    # model = torch.hub.load('pytorch/vision:v0.11.3', 'deeplabv3_resnet50', pretrained=True)

    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True
    )
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
