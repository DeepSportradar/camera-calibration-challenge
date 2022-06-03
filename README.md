[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm) [![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/1685/overview) [![Win 2x $500](https://badgen.net/badge/win/2x%20%24500/yellow)](http://mmsports.multimedia-computing.de/mmsports2022/challenge.html) [![Kaggle Dataset](https://badgen.net/badge/kaggle/dataset/blue)](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset)

# Camera Calibration Challenge

Maintainer: Davide Zambrano from Sportradar (d.zambrano@sportradar.com)

We present the "Camera Calibration Challenge" for ACM MMSports 2022 the 5th International ACM Workshop on Multimedia Content Analysis in Sports. This year, MMSports proposes a competition where participants will compete over State-of-the-art problems applied to real-world sport specific data. The competition is made of 4 individual challenges, each of which is sponsored by [Sportradar](https://www.sportradar.com) with a $1'000.00 prize.

The "Camera Calibration Challenge" aims at predicting the camera calibration parameters from images taken from basketball games. Please refer to [Challenge webpage](https://deepsportradar.github.io/challenge.html) for the general challenge rules.

This repo is based on the [Pytorch Project Template](https://github.com/L1aoXingyu/Deep-Learning-Project-Template). We want to thank the authors for providing this tool, please refer to the original repo for the full documentation. This version applies some changes to the original code to specifically adapt it to the "Camera Calibration Challenge" for ACM MMSports 2022.

# Table Of Contents

- [In a Nutshell](#in-a-nutshell)
- [Installation](#installation)
- [Requirements](#requirements)
- [In Details](#in-details)

  - [Download and prepare the dataset](#download-and-prepare-the-dataset)
  - [Challenge rules](#challenge-rules)
  - [The Baseline](#the-baseline)

    - [Training the segmentation model](#training-the-segmentation-model)
    - [Evaluate the camera calibration model](#evaluate-the-camera-calibration-model)

  - [Submission format](#submission-format)

- [Acknowledgments](#acknowledgments)

# In a Nutshell

The purpose of this challenge is to predict the camera calibration parameters from a single frame of a basketball game. Participants have access to a dataset of 728 pairs of images and camera calibration parameters. By default these pairs are divided in train (548), val (96) and test (84) splits. Note that this test split is different from the one on which the challenge participants will be evaluated on. Therefore, all the 728 examples can be used for the training purpose.

Participants are encouraged to explore different methods to predict the camera calibration parameters. However, a baseline will be provided as described in the [In Details](#in-details) section.

Predictions will be evaluated based on a [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) of the projection error of 6 points--left, center and right extremities at the middle and bottom parts of the frame--in the 3D coordinates.

# Installation

A convenience [bash script](./install.sh) is provided that sets up the python environment needed to run the camera-calibration-challenge project.

The script will try to install the library into a _conda_ environment alongside with all dependencies. The _conda_ environment name is defaulted to `camera-calibration`, but can be overridden by the user:

```sh
./install.sh [my-conda-env]
```

Otherwise, please make sure to install the proper requirements.

# Requirements

As in the original repo, this project relies on:

- [yacs](https://github.com/rbgirshick/yacs) (Yet Another Configuration System)
- [PyTorch](https://pytorch.org/) (An open source deep learning platform)
- [ignite](https://github.com/pytorch/ignite) (High-level library to help with training neural networks in PyTorch)

Moreover, data are handled by:

- [deepsport-utilities](https://gitlab.com/deepsport/deepsport_utilities) (A Dataset API)

# In Details

## Download and prepare the dataset

The dataset can be found [here](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset). It can be downloaded and unzipped manually in the `basketball-instants-dataset/` folder of the project.

We will here download it programmatically. First install the kaggle CLI.

```bash
pip install kaggle
```

Go to your Kaggle Account page and click on `Create new API Token` to download the file to be saved as `~/.kaggle/kaggle.json` for authentication.

```bash
kaggle datasets download deepsportradar/basketball-instants-dataset
mkdir basketball-instants-dataset
unzip -qo ./basketball-instants-dataset.zip -d basketball-instants-dataset
```

The dataset has to be pre-processed to be used, please run:

```bash
python tools/download_dataset.py --dataset-folder ./basketball-instants-dataset --output-folder dataset
```

The processed dataset is then contained in a `pickle` file in the `dataset` folder. Please refer to `.data\datasets\viewds.py` methods as examples of usage. Specifically the class `GenerateSViewDS` applies the required transformations and splits the keys into `train`, `val` and `test`. Please consider that the `test` keys of this dataset are not the ones used for the challenge evaluation (those keys, without annotations, will be provided in a second phase of the challenge). The class `SVIEWDS` is an example of `torch.utils.data.Dataset` for PyTorch users. Finally, note that transformations are applied at each query of the key, thus returning a potentially infinite pairs of image (views) and calibration matrix. A pseudo-random transformation is applied for the `val` and `test` keys, thus views are fixed for these splits.

The challenge uses the split defined by [`DeepSportDatasetSplitter`](https://github.com/DeepSportRadar/camera-calibration-challenge/blob/0d75313576055f67ac9b5cc999e4a9f91ae90e12/data/datasets/viewds.py#L221) which

1. Uses images from `KS-FR-CAEN`, `KS-FR-LIMOGES` and `KS-FR-ROANNE` arenas for the **testing-set**.
2. Randomly samples 15% of the remaining images for the **validation-set**
3. Uses the remaining images for the **training-set**.

The **testing-set** should be used to evaluate your model, both on the public EvalAI leaderboard that provides the temporary ranking, and when communicating about your method.

The **challenge-set** will be shared later, without the labels, and will be used for the official ranking. You are free to use the three sets defined above to build the final model on which your method will be evaluated in the EvalAI submission.

Each key in the dataset is associated with an item which contains the images to be used as input and the Calib object from [calib3d](https://github.com/ispgroupucl/calib3d) library, which is what participants should predict.

Images are created as views of basketball games from the original cameras of the Keemotion system. These images can be considered as single frames of a broadcasted basketball game. Indeed, the view creation takes into account the location of the ball, and, in basketball, most of the action is around the KEY area under the rim (you can look at the [Basketball court](https://en.wikipedia.org/wiki/Basketball_court#Table) page and the `utils/intersections.py` file for some definitions). All the games in this dataset are from FIBA courts. In this challenge we consider un-distorted images only. Camera conventions are described [here](https://gitlab.com/deepsport/deepsport_utilities/-/blob/main/calibration.md).

The Calib object is built around the K (calibration), T (translation) and R (rotation) matrixes (reference [Camera matrix](https://en.wikipedia.org/wiki/Camera_matrix))

## Challenge rules

The challenge goal is to obtain the lowest MSE (cm) on images that were not seen during training. In particular, the leaderboards that provide rewards will be built on an unannotated challenge set that will be provided late in June.

The competitors are asked to create models that only rely on the provided data for training. (except for initial weights that can come from well-established public methods pre-trained on public data. This must be clearly stated in publication/report)

Please see the challenge page for more details: <https://deepsportradar.github.io/challenge.html>.

## The Baseline

We encourage participants to find innovative solutions to solve the camera calibration challenge. However, an initial baseline is provided as example. The baseline is composed by two models: the first is a segmentation model that predicts the 20 lines of the basketball court (`DeepLabv3` in `modeling/example_model.py`); the second finds the 2D intersections in the image space and matches them with the visible 3D locations of the court (see `utils/intersections.py`). If enough intersections points are found (>5) the method `cv2.calibrateCamera` predicts the camera parameters (see `compute_camera_model` in `modeling/example_camera_model.py`). In all the other cases, the model returns an average of the camera parameters in the training set as default.

### Training the segmentation model

Once the dataset is downloaded (see [Download and prepare the dataset](#download-and-prepare-the-dataset)), you can train the baseline by running:

```python
python tools/train_net.py --config_file configs/train_sviewds_public_baseline.yml
```

Logs and weights will be saved in the `OUTPUT_DIR` folder specified in the config file; moreover, [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) events will be saved in the `runs` folder. After the training, you can test the segmentation model using the same config file specifying the `TEST.WEIGHT` path in the parameters:

```python
python tools/test_net.py --config_file configs/train_sviewds_public_baseline.yml
```

The config parameter `DATASET.EVAL_ON` specifies on which split the model will be tested: `val` or `test`.

### Evaluate the camera calibration model

The script `evaluate_net.py` runs the inference on the trained model and generates the `predictions.json` file as required for the submission (see [Submission format](#submission-format)).

```python
python tools/evaluate_net.py --config_file configs/eval_sviewds_public_baseline.yml
```

Note that the config file used for camera calibration model evaluation is different from the one used for training and testing the segmentation model only. Here, the config parameter `DATASET.EVAL_ON` specifies on which split the model will be tested: `val` or `test`. During evaluation, the dataloader will return now as target the segmentation mask and the groundtruth camera calibration parameters. As explained before, the predicted camera parameters will be computed based on the intersections found with the `find_intersections` method in `utils/intersections.py` and the `compute_camera_model` method in `modeling/example_camera_model.py`. Once the predictions have been saved, if the flag `DATASETS.RUN_METRICS` is set as `True`, the method `run_metrics` in `engine/example_evaluation.py` will compare the camera calibration parameters with the corresponding groundtruth camera calibration parameters (`val` or `test`). Please consider that the `test` keys are not the ones used for the challenge evaluation (those keys, without annotations, will be provided in a second phase of the challenge).

You can now submit the `predictions.json` on EvalAI for the `Test` phase and verify that the results are the same.

When the challenge set will be released, you will need to set `DATASETS.RUN_METRICS` as `False` and generate the prediction only.

## Submission format

The submission format is a single `json` file containing a list of dicts. Each dict should contain all the camera parameters `T`, `K`, `kc`, `R`, `C`, `P`, `Pinv`, `Kinv`. Note that the evaluation script retrieves the camera parameters from the Projection Matrix `P`. See the class [calib3d.Calib](https://github.com/ispgroupucl/calib3d/blob/b20694a42a3e043b157dcd9b363833184cc3fcdc/calib3d/calib.py#L155). Please consider that the evaluation script follows the list of images provided: an empty dict will be replaced by a diagonal homography (see `run_metrics` in `engine/example_evaluation.py`).

Once the camera model is provided, the evaluation script projects 6 points from the image space to the 3D coordinates. On these projections the mean squared error is computed.

The prediction file has to be submitted at the [EvalAI](https://eval.ai/web/challenges/challenge-page/1687/overview) page of the challenge.

# Acknowledgments

[Sportradar](https://sportradar.com/) and collaborators:

- Gabriel Van Zandycke (g.vanzandycke@sportradar.com)
- Maxime Istasse (m.istasse@sportradar.com)
- Vladimir Somers (v.somers@sportradar.com)
