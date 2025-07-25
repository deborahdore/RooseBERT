#!/bin/bash
#SBATCH --job-name=llama_ner

#SBATCH -C a100
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=3

#SBATCH --time=20:00:00
#SBATCH --output=logs/llama_ner_%j.out
#SBATCH --error=logs/llama_ner_%j.out

#SBATCH --hint=nomultithread

module purge
module load arch/a100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate roosebert

export TOKENIZERS_PARALLELISM=false


for i in {1..5}
do
  echo "------------------------> Run #$i"
  echo "ner"
  python src/fs_ner.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
done