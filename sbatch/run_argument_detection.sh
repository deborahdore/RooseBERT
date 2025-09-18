#!/bin/bash
set -e
set -u

module purge
module load miniconda
conda activate rooseBERT

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="argument_detection"

wandb offline
wandb disabled

# HYPERPARAMETERS -----------------------------
MODEL=""
MODEL_DIR=""

LEARNING_RATES=(2e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01)
BATCH_SIZES=(8 16 32)
MAX_LENGTHS=(512)
EPOCHS=(2 3 4)
# ---------------------------------------------

for lr in "${LEARNING_RATES[@]}"; do
  for wd in "${WEIGHT_DECAYS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
      for max_length in "${MAX_LENGTHS[@]}"; do
        for epoch in "${EPOCHS[@]}"; do

          RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s-ML%s" "$MODEL" "$epoch" "$lr" "$wd" "$batch" "$max_length")
          OUTPUT_DIR="./logs/$WANDB_PROJECT/$MODEL/$RUN_NAME"

          mkdir -p "$OUTPUT_DIR"

          python src/run_ner.py \
            --run_name "$RUN_NAME" \
            --model_name_or_path "$MODEL_DIR/$MODEL" \
            --config_name "$MODEL_DIR/$MODEL" \
            --tokenizer_name "$MODEL_DIR/$MODEL" \
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
            --logging_steps 500 \
            --save_strategy "epoch" \
            --save_total_limit 1 \
            --seed 42 \
            --report_to "wandb" \
            --eval_on_start \
            --remove_unused_columns
        done
      done
    done
  done
done