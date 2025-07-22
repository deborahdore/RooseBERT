#!/bin/bash
#SBATCH --job-name=RooseBERT-random-seed
#SBATCH --output=logs/RooseBERT_%j.out
#SBATCH --gres=gpu:1

set -e
set -u

wandb disabled
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="argument_detection"

SEED=(42 2025 77 206 0)
rm -rf /home/ddore/workspace/5Runs/logs/argument_detection/RooseBERT-src-*
# ----------------------------------------------------------------------------------------------------------------------
#model="RooseBERT-cont-cased"
#epoch=4
#batch=8
#lr=5e-5
#max_length=512
#wd=0.01
#
#for seed in "${SEED[@]}"; do
#  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
#  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  python src/run_ner.py \
#    --run_name "$RUN_NAME" \
#    --model_name_or_path "/home/ddore/workspace/5Runs/models/$model" \
#    --config_name "/home/ddore/workspace/5Runs/models/$model" \
#    --tokenizer_name "/home/ddore/workspace/5Runs/models/$model" \
#    --cache_dir "./cache/" \
#    --logging_dir "./logs" \
#    --output_dir "$OUTPUT_DIR" \
#    --train_file "./data/argument_detection/train.json" \
#    --validation_file "./data/argument_detection/dev.json" \
#    --test_file "./data/argument_detection/test.json" \
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
#    --remove_unused_columns
#  done
#
#model="RooseBERT-cont-uncased"
#epoch=4
#batch=8
#lr=5e-5
#max_length=256
#wd=0.1
#
#for seed in "${SEED[@]}"; do
#  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
#  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  python src/run_ner.py \
#    --run_name "$RUN_NAME" \
#    --model_name_or_path "/home/ddore/workspace/5Runs/models/$model" \
#    --config_name "/home/ddore/workspace/5Runs/models/$model" \
#    --tokenizer_name "/home/ddore/workspace/5Runs/models/$model" \
#    --cache_dir "./cache/" \
#    --logging_dir "./logs" \
#    --output_dir "$OUTPUT_DIR" \
#    --train_file "./data/argument_detection/train.json" \
#    --validation_file "./data/argument_detection/dev.json" \
#    --test_file "./data/argument_detection/test.json" \
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
#    --remove_unused_columns
#  done

#model="RooseBERT-scr-cased"
#epoch=4
#batch=8
#lr=5e-5
#max_length=512
#wd=0.1
#SEED=(42 2025 77 206 0 102 451 168 913)
#
#for seed in "${SEED[@]}"; do
#  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
#  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#  mkdir -p "$OUTPUT_DIR"
#
#  python src/run_ner.py \
#    --run_name "$RUN_NAME" \
#    --model_name_or_path "/home/ddore/workspace/5Runs/models/$model" \
#    --config_name "/home/ddore/workspace/5Runs/models/$model" \
#    --tokenizer_name "/home/ddore/workspace/5Runs/models/$model" \
#    --cache_dir "./cache/" \
#    --logging_dir "./logs" \
#    --output_dir "$OUTPUT_DIR" \
#    --train_file "./data/argument_detection/train.json" \
#    --validation_file "./data/argument_detection/dev.json" \
#    --test_file "./data/argument_detection/test.json" \
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
#    --remove_unused_columns
#  done

model="RooseBERT-scr-uncased"
epoch=4
batch=8
lr=5e-5
max_length=512
wd=0.1
SEED=(168 333 960 713 14)
for seed in "${SEED[@]}"; do
  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"

  mkdir -p "$OUTPUT_DIR"

  python src/run_ner.py \
    --run_name "$RUN_NAME" \
    --model_name_or_path "/home/ddore/workspace/5Runs/models/$model" \
    --config_name "/home/ddore/workspace/5Runs/models/$model" \
    --tokenizer_name "/home/ddore/workspace/5Runs/models/$model" \
    --cache_dir "./cache/" \
    --logging_dir "./logs" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "./data/argument_detection/train.json" \
    --validation_file "./data/argument_detection/dev.json" \
    --test_file "./data/argument_detection/test.json" \
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
    --remove_unused_columns
  done