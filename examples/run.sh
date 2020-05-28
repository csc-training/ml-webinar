#!/bin/bash
#SBATCH --account=project_2002586
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:1

module load tensorflow/2.0.0
srun python3 dogs_vs_cats.py

