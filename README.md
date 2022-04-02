# CHALLENGE START OFFICIALLY APRIL 6th. Stayed tuned.


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
