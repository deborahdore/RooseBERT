#!/bin/bash
#SBATCH --job-name=gemma_argument_detection

#SBATCH -C a100
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3

#SBATCH --time=20:00:00
#SBATCH --output=logs/gemma_argument_detection_%j.out
#SBATCH --error=logs/gemma_argument_detection_%j.out

#SBATCH --hint=nomultithread

module purge
module load arch/a100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate roosebert

export TOKENIZERS_PARALLELISM=false

python src/fs_argument_detection.py --model_id "google/gemma-3-4b-it"