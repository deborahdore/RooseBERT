#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false

MODEL="mistralai/Mistral-7B-Instruct-v0.3"
DATASET="ParlVote+"
TASK="multi_class_classification"
echo $TASK

python src_llm/zeroshot/zs_multiclass_classification.py --model $MODEL --dataset $DATASET
python src_llm/fewshot/fs_multiclass_classification.py --model $MODEL --dataset $DATASET
python src_llm/fine-tuning/ft_multiclass_classification.py --model $MODEL --dataset $DATASET

