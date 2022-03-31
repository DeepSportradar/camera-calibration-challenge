#!/bin/sh

RED='\033[0;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m' # No Color

BUILD_DIR="/tmp/homography_install"

# Make sure cuda is available
if ! [ -x "$(command -v nvidia-smi)" ]; then
  echo $RED"Cuda not found"$NC
  exit 1
else
  echo $GREEN"Cuda found"$NC
fi

# Miniconda installation
if ! [ -x "$(command -v conda)" ]; then
    echo $CYAN"Installing miniconda"$NC
    miniconda_filename="${BUILD_DIR}/Miniconda3-py39_4.10.3-Linux-x86_64.sh"
    curl https://repo.anaconda.com/miniconda/$miniconda_filename --output $miniconda_filename
    bash $miniconda_filename

    if ! [ -x "$(command -v conda)" ]; then
        echo $RED"Failed to insatll miniconda, aborting"$NC
        exit 1
    else
        echo $GREEN"Miniconda succesfully installed"$NC
        rm $miniconda_filename
    fi
else
    anaconda_path=$(which conda)
    echo $GREEN"Anaconda found at $anaconda_path"$NC
fi


# Python environment setup
conda_env_name=${1:-minimap}
if ! conda info --envs | grep $conda_env_name; then
    echo $CYAN"Creating conda environment '$conda_env_name'"$NC
    conda create -n $conda_env_name python=3.8
fi

echo
echo $CYAN"Activating environment '$conda_env_name'"$NC
eval "$(conda shell.bash hook)"
conda activate $conda_env_name

if ! conda info --envs | grep $conda_env_name; then
    echo $RED"An error occurred while activating conda environment '$conda_env_name'"$NC
    exit 1
else
    echo $GREEN"Successfully activated conda environment '$conda_env_name'"$NC
fi

# Git submodules
echo
echo $CYAN"Updating git submodules"$NC
git submodule update --init --recursive

echo $GREEN"Success"$NC

# Pytorch, TorchVision and CudaToolkit
echo
echo $CYAN"Installing Pytorch, TorchVision and CudaToolkit"$NC
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

echo $GREEN"Success"$NC

# Minimap installation
echo
echo $CYAN"Installing homography project in edit mode with associated dependencices"$NC
pip install -r requirements.txt
pip install -e .

echo $GREEN"Success"$NC

echo
echo $GREEN"Environment setup completed"$NC
echo $GREEN"Activate the conda environment '$conda_env_name' with 'conda activate $conda_env_name'"$NC

