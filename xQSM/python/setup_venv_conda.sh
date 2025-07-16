#!/bin/bash

#$ -N xQSM_venv_setup
#$ -o /2025-Summer-Research/xQSM/python/job_info
#$ -e /2025-Summer-Research/xQSM/python/job_info
#$ -wd /home/zcemska/Scratch/DeepLearningQSM

# Script to create a conda virtual environment for xQSM training
ENV_NAME="QSM"
PYTHON_VERSION="3.9"

echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"

# Create the conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA support..."
# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8 (as shown in your shell script)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install nibabel scipy matplotlib jupyter torchinfo glob2
