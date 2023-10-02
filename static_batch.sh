#!/bin/bash
#SBATCH -J static_trajectory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH -t 100:00:00

source activate grid38

python -B main_trajectory.py