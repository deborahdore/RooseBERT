#!/bin/bash
set -e
conda activate roosebert
export TOKENIZERS_PARALLELISM=false

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
DATASET="ParlVote"
TASK="binary_classification"
echo $TASK

python src_llm/zeroshot/zs_binary_classification.py --model $MODEL --dataset $DATASET
python src_llm/fewshot/fs_binary_classification.py --model $MODEL --dataset $DATASET
python src_llm/fine-tuning/ft_binary_classification.py --model $MODEL --dataset $DATASET

