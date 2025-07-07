#!/bin/bash
# RUN THIS SCRIPT USING THE RUN_NER AND RUN_CLASSIFICATION OF THE 'exp-polibertweet' BRANCH
set -e
set -u

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="PoliBERTweet-relation_classification"
export WANDB_WATCH="all"

# HYPERPARAMETERS -----------------------------
MODELS=("polibertweet-mlm")
LEARNING_RATES=(2e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.1 0.01)
BATCH_SIZES=(8 16 32)
MAX_LENGTHS=(128 256 512)
EPOCHS=(2 3 4 6)
# ---------------------------------------------

for model in "${MODELS[@]}"; do
  for lr in "${LEARNING_RATES[@]}"; do
    for wd in "${WEIGHT_DECAYS[@]}"; do
      for batch in "${BATCH_SIZES[@]}"; do
        for max_length in "${MAX_LENGTHS[@]}"; do
          for epoch in "${EPOCHS[@]}"; do

            RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$model" "$epoch" "$lr" "$wd" "$batch" "$max_length")
            OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"

            mkdir -p "$OUTPUT_DIR"

            python src/run_classification.py \
              --run_name "$RUN_NAME" \
              --model_name_or_path "kornosk/$model" \
              --config_name "kornosk/$model" \
              --tokenizer_name "kornosk/$model" \
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
              --save_total_limit 5 \
              --seed 42 \
              --report_to "wandb" \
              --text_column_name "text" \
              --label_column_name "link_type" \
              --eval_on_start \
              --remove_unused_columns
          done
        done
      done
    done
  done
done
