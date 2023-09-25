#!/bin/bash
#SBATCH -J static_batch
#SBATCH -t 100:00:00

source activate grid38

python -B main_trajectory.py