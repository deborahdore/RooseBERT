#!/bin/bash
#SBATCH --job-name=bert-large-random-seed
#SBATCH --output=logs/bert-large_%j.out
#SBATCH --gres=gpu:1

set -e
set -u

wandb disabled
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="relation_classification"

SEED=(168 113)
# ----------------------------------------------------------------------------------------------------------------------
#model="bert-large-cased"
#epoch=3
#batch=16
#lr=2e-5
#max_length=512
#wd=0.01
#
#for seed in "${SEED[@]}"; do
#  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
#  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  python src/run_classification.py \
#    --run_name "$RUN_NAME" \
#    --model_name_or_path "$model" \
#    --config_name "$model" \
#    --tokenizer_name "$model" \
#    --cache_dir "./cache/" \
#    --logging_dir "./logs" \
#    --output_dir "$OUTPUT_DIR" \
#    --train_file "./data/relation_classification/train.csv" \
#    --validation_file "./data/relation_classification/dev.csv" \
#    --test_file "./data/relation_classification/test.csv" \
#    --eval_strategy "steps" \
#    --eval_steps 1000 \
#    --per_device_train_batch_size "$batch" \
#    --per_device_eval_batch_size "$batch" \
#    --learning_rate "$lr" \
#    --max_seq_length "$max_length" \
#    --weight_decay "$wd" \
#    --num_train_epochs "$epoch" \
#    --logging_strategy "steps" \
#    --logging_steps 100 \
#    --save_strategy "epoch" \
#    --save_total_limit 1 \
#    --seed "$seed" \
#    --report_to "wandb" \
#    --eval_on_start \
#    --remove_unused_columns \
#    --text_column_name "text" \
#    --label_column_name "link_type"
#  done


model="bert-large-uncased"
epoch=3
batch=32
lr=3e-5
max_length=512
wd=0.01

for seed in "${SEED[@]}"; do
  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"

  mkdir -p "$OUTPUT_DIR"

  python src/run_classification.py \
    --run_name "$RUN_NAME" \
    --model_name_or_path "$model" \
    --config_name "$model" \
    --tokenizer_name "$model" \
    --cache_dir "./cache/" \
    --logging_dir "./logs" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "./data/relation_classification/train.csv" \
    --validation_file "./data/relation_classification/dev.csv" \
    --test_file "./data/relation_classification/test.csv" \
    --eval_strategy "steps" \
    --eval_steps 1000 \
    --per_device_train_batch_size "$batch" \
    --per_device_eval_batch_size "$batch" \
    --learning_rate "$lr" \
    --max_seq_length "$max_length" \
    --weight_decay "$wd" \
    --num_train_epochs "$epoch" \
    --logging_strategy "steps" \
    --logging_steps 100 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed "$seed" \
    --report_to "wandb" \
    --eval_on_start \
    --remove_unused_columns \
    --text_column_name "text" \
    --label_column_name "link_type"
  done