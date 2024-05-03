#!/bin/bash

#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --output=run_a100_1.out

eval "$(conda shell.bash hook)"

conda activate project

python3 train_test.py cnn --GPU --tune both