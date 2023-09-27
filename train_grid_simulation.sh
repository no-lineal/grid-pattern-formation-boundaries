#!/bin/bash
#SBATCH -J grid_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40g:4
#SBATCH -t 100:00:00

source activate grid38

python -B main_trajectory.py
python -B main.py