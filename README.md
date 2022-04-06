# Camera Calibration Challenge

Mantainer: Davide Zambrano from Sportradar (d.zambrano@sportradar.com)

We present the "Camera Calibration Challenge" for ACM MMSports 2022 the 5th International ACM Workshop on Multimedia Content Analysis in Sports. This year, MMSports proposes a competition where participants will compete over State-of-the-art problems applied to real-world sport specific data. The competition is made of 4 individual challenges, each of which is sponsored by Sportradar with a $1'000.00 prize.

The "Camera Calibration Challenge" aims at predicting the camera calibration parameters from images taken from basketball games.

This repo is based on the [Pytorch Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template). We want to thank the authors for providing this tool, please refer to the original repo for the full documentation. This version applies some changes to the original code to specifically adapt it to the "Camera Calibration Challenge" for ACM MMSports 2022.

# Table Of Contents

- [In a Nutshell](#in-a-nutshell)
- [Installation](#installation)
- [Requirements](#requirements)
- [In Details](#in-details)
- [Acknowledgments](#acknowledgments)

# In a Nutshell

The purpose of this challenge is to predict the camera calibration parameters from a single frame of a basketball game. Participants have access to a dataset of 728 pairs of images and camera calibration parameters. By default these pairs are devided in train (548), val (96) and test (84) splits. Note that this test split is different from the one on which the challenge participants will be evaluated on. Therefore, all the 728 examples can be used for the training purpose.

Participants are encuraged to explore different methods to predict the camera calibration parameters. However, a baseline will be provided as described in the [In Details](#in-details) section.

Predictions will be evaluated based on a [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) of the projection error of 6 points--left, center and right extremities at the middle and bottom parts of the frame--in the 3D coordinates.

# Installation

A convenience [bash script](./install.sh) is provided that sets up the python environment needed to run the camera-calibration-challenge project.

The script will try to install the library into a _conda_ environment alongside with all dependencies. The _conda_ evironment name is defaulted to `camera-calibration`, but can be overridden by the user:

```sh
./install.sh [my-conda-env]
```

Otherwise, please make sure to install the proper requirements.

# Requirements

As in the original repo, this project relies on:

- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform)
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

Moreover, data are handeled by:

- [deepsport-utilities](https://gitlab.com/deepsport/deepsport_utilities) (A Dataset API)

# In Details

# Acknowledgments
