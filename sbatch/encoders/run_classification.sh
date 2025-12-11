#!/bin/bash
#SBATCH --job-name=binary_AusHansard
#SBATCH --gres=gpu:1

#SBATCH --output=logs/binary_AusHansard.out
#SBATCH --error=logs/binary_AusHansard.out
set -e
set -u

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="binary_classification_AusHansard"

wandb offline
wandb disabled

export HF_HOME="/home/ddore/.cache/huggingface"

## HYPERPARAMETERS bert -----------------------------
#MODEL_DIR="google-bert"
#MODELS=("bert-base-uncased" "bert-base-cased")
#LEARNING_RATES=(2e-5 3e-5 5e-5)
#WEIGHT_DECAYS=(0.01)
#BATCH_SIZES=(8 16 32)
#EPOCHS=(2 3 4)
## ---------------------------------------------
#for model in "${MODELS[@]}"; do
#  for lr in "${LEARNING_RATES[@]}"; do
#    for wd in "${WEIGHT_DECAYS[@]}"; do
#      for batch in "${BATCH_SIZES[@]}"; do
#        for epoch in "${EPOCHS[@]}"; do
#
#          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
#          OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#          mkdir -p "$OUTPUT_DIR"
#
#          python src/run_classification.py \
#            --run_name "$RUN_NAME" \
#            --model_name_or_path "$MODEL_DIR/$model" \
#            --config_name "$MODEL_DIR/$model" \
#            --tokenizer_name "$MODEL_DIR/$model" \
#            --cache_dir "./cache/" \
#            --logging_dir "./logs/" \
#            --output_dir "$OUTPUT_DIR" \
#            --train_file "./data/binary_classification/AusHansard/train.csv" \
#            --validation_file "./data/binary_classification/AusHansard/dev.csv" \
#            --test_file "./data/binary_classification/AusHansard/test.csv" \
#            --eval_strategy "steps" \
#            --eval_steps 1500 \
#            --per_device_train_batch_size "$batch" \
#            --per_device_eval_batch_size "$batch" \
#            --learning_rate "$lr" \
#            --weight_decay "$wd" \
#            --num_train_epochs "$epoch" \
#            --logging_strategy "steps" \
#            --logging_steps 1000 \
#            --save_strategy "epoch" \
#            --save_total_limit 1 \
#            --seed 42 \
#            --report_to "wandb" \
#            --text_column_name "text" \
#            --label_column_name "label" \
#            --eval_on_start \
#            --remove_unused_columns
#        done
#      done
#    done
#  done
#done
#
## HYPERPARAMETERS conflibert -----------------------------
#MODEL_DIR="snowood1"
#MODELS=("ConfliBERT-scr-uncased" "ConfliBERT-scr-cased" "ConfliBERT-cont-uncased" "ConfliBERT-cont-cased")
#LEARNING_RATES=(2e-5 3e-5 5e-5)
#WEIGHT_DECAYS=(0.01)
#BATCH_SIZES=(8 16 32)
#EPOCHS=(2 3 4)
## ---------------------------------------------
#for model in "${MODELS[@]}"; do
#  for lr in "${LEARNING_RATES[@]}"; do
#    for wd in "${WEIGHT_DECAYS[@]}"; do
#      for batch in "${BATCH_SIZES[@]}"; do
#        for epoch in "${EPOCHS[@]}"; do
#
#          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
#          OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#          mkdir -p "$OUTPUT_DIR"
#
#          python src/run_classification.py \
#            --run_name "$RUN_NAME" \
#            --model_name_or_path "$MODEL_DIR/$model" \
#            --config_name "$MODEL_DIR/$model" \
#            --tokenizer_name "$MODEL_DIR/$model" \
#            --cache_dir "./cache/" \
#            --logging_dir "./logs/" \
#            --output_dir "$OUTPUT_DIR" \
#            --train_file "./data/binary_classification/AusHansard/train.csv" \
#            --validation_file "./data/binary_classification/AusHansard/dev.csv" \
#            --test_file "./data/binary_classification/AusHansard/test.csv" \
#            --eval_strategy "steps" \
#            --eval_steps 1500 \
#            --per_device_train_batch_size "$batch" \
#            --per_device_eval_batch_size "$batch" \
#            --learning_rate "$lr" \
#            --weight_decay "$wd" \
#            --num_train_epochs "$epoch" \
#            --logging_strategy "steps" \
#            --logging_steps 1000 \
#            --save_strategy "epoch" \
#            --save_total_limit 1 \
#            --seed 42 \
#            --report_to "wandb" \
#            --text_column_name "text" \
#            --label_column_name "label" \
#            --eval_on_start \
#            --remove_unused_columns
#        done
#      done
#    done
#  done
#done
## HYPERPARAMETERS polibert -----------------------------
#MODEL_DIR="kornosk"
#MODELS=("polibertweet-political-twitter-roberta-mlm")
#LEARNING_RATES=(2e-5 3e-5 5e-5)
#WEIGHT_DECAYS=(0.01)
#BATCH_SIZES=(8 16 32)
#EPOCHS=(2 3 4)
## ---------------------------------------------
#for model in "${MODELS[@]}"; do
#  for lr in "${LEARNING_RATES[@]}"; do
#    for wd in "${WEIGHT_DECAYS[@]}"; do
#      for batch in "${BATCH_SIZES[@]}"; do
#        for epoch in "${EPOCHS[@]}"; do
#
#          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
#          OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#          mkdir -p "$OUTPUT_DIR"
#
#          python src/run_classification.py \
#            --run_name "$RUN_NAME" \
#            --model_name_or_path "$MODEL_DIR/$model" \
#            --config_name "$MODEL_DIR/$model" \
#            --tokenizer_name "$MODEL_DIR/$model" \
#            --cache_dir "./cache/" \
#            --logging_dir "./logs/" \
#            --output_dir "$OUTPUT_DIR" \
#            --train_file "./data/binary_classification/AusHansard/train.csv" \
#            --validation_file "./data/binary_classification/AusHansard/dev.csv" \
#            --test_file "./data/binary_classification/AusHansard/test.csv" \
#            --eval_strategy "steps" \
#            --eval_steps 1500 \
#            --per_device_train_batch_size "$batch" \
#            --per_device_eval_batch_size "$batch" \
#            --learning_rate "$lr" \
#            --weight_decay "$wd" \
#            --num_train_epochs "$epoch" \
#            --logging_strategy "steps" \
#            --logging_steps 1000 \
#            --save_strategy "epoch" \
#            --save_total_limit 1 \
#            --seed 42 \
#            --report_to "wandb" \
#            --text_column_name "text" \
#            --label_column_name "label" \
#            --eval_on_start \
#            --remove_unused_columns
#        done
#      done
#    done
#  done
#done
#
## HYPERPARAMETERS roberta -----------------------------
#MODEL_DIR="FacebookAI"
#MODELS=("roberta-base")
#LEARNING_RATES=(2e-5 3e-5 5e-5)
#WEIGHT_DECAYS=(0.01)
#BATCH_SIZES=(8 16 32)
#EPOCHS=(2 3 4)
## ---------------------------------------------
#for model in "${MODELS[@]}"; do
#  for lr in "${LEARNING_RATES[@]}"; do
#    for wd in "${WEIGHT_DECAYS[@]}"; do
#      for batch in "${BATCH_SIZES[@]}"; do
#        for epoch in "${EPOCHS[@]}"; do
#
#          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
#          OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
#
#          mkdir -p "$OUTPUT_DIR"
#
#          python src/run_classification.py \
#            --run_name "$RUN_NAME" \
#            --model_name_or_path "$MODEL_DIR/$model" \
#            --config_name "$MODEL_DIR/$model" \
#            --tokenizer_name "$MODEL_DIR/$model" \
#            --cache_dir "./cache/" \
#            --logging_dir "./logs/" \
#            --output_dir "$OUTPUT_DIR" \
#            --train_file "./data/binary_classification/AusHansard/train.csv" \
#            --validation_file "./data/binary_classification/AusHansard/dev.csv" \
#            --test_file "./data/binary_classification/AusHansard/test.csv" \
#            --eval_strategy "steps" \
#            --eval_steps 1500 \
#            --per_device_train_batch_size "$batch" \
#            --per_device_eval_batch_size "$batch" \
#            --learning_rate "$lr" \
#            --weight_decay "$wd" \
#            --num_train_epochs "$epoch" \
#            --logging_strategy "steps" \
#            --logging_steps 1000 \
#            --save_strategy "epoch" \
#            --save_total_limit 1 \
#            --seed 42 \
#            --report_to "wandb" \
#            --text_column_name "text" \
#            --label_column_name "label" \
#            --eval_on_start \
#            --remove_unused_columns
#        done
#      done
#    done
#  done
#done

# HYPERPARAMETERS deberta -----------------------------
MODEL_DIR="microsoft"
MODELS=("deberta-base")
LEARNING_RATES=(2e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01)
BATCH_SIZES=(8 16 32)
EPOCHS=(2 3 4)
# ---------------------------------------------
for model in "${MODELS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      for batch in "${BATCH_SIZES[@]}"; do
        for epoch in "${EPOCHS[@]}"; do

          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
          OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"

          mkdir -p "$OUTPUT_DIR"

          python src/run_classification.py \
            --run_name "$RUN_NAME" \
            --model_name_or_path "$MODEL_DIR/$model" \
            --config_name "$MODEL_DIR/$model" \
            --tokenizer_name "$MODEL_DIR/$model" \
            --cache_dir "./cache/" \
            --logging_dir "./logs/" \
            --output_dir "$OUTPUT_DIR" \
            --train_file "./data/binary_classification/AusHansard/train.csv" \
            --validation_file "./data/binary_classification/AusHansard/dev.csv" \
            --test_file "./data/binary_classification/AusHansard/test.csv" \
            --eval_strategy "steps" \
            --eval_steps 1500 \
            --per_device_train_batch_size "$batch" \
            --per_device_eval_batch_size "$batch" \
            --learning_rate "$lr" \
            --weight_decay "$wd" \
            --num_train_epochs "$epoch" \
            --logging_strategy "steps" \
            --logging_steps 1000 \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --seed 42 \
            --report_to "wandb" \
            --text_column_name "text" \
            --label_column_name "label" \
            --eval_on_start \
            --remove_unused_columns
        done
      done
    done
  done
done