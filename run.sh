#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-gpu=80g
#SBATCH --array=0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=3

conda activate sok

python prepare_dataset.py

# python open_models/training.py open_models/train.json

