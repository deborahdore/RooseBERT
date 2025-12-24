#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false

MODEL="google/gemma-3-4b-it"
DATASET="ArgUNSC"
TASK="sequence_labelling"
echo $TASK

python src_llm/zeroshot/zs_sequence_labelling.py --model $MODEL --dataset $DATASET
python src_llm/fewshot/fs_sequence_labelling.py --model $MODEL --dataset $DATASET
python src_llm/fine-tuning/ft_sequence_labelling.py --model $MODEL --dataset $DATASET