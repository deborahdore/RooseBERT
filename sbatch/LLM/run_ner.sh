#!/bin/bash
conda activate roosebert

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="Large-Language-Models_ner"
wandb disabled

# meta-llama/Meta-Llama-3-8B
# mistralai/Mistral-7B-v0.3
# google/gemma-2-9b

MODEL_PATH="mistralai"
MODEL="Mistral-7B-v0.3"
N_GPUS=2

# HYPERPARAMETERS -----------------------------
LEARNING_RATES=(2e-5 3e-5 5e-5)
WEIGHT_DECAYS=(0.01)
BATCH_SIZES=(8 16 32)
MAX_LENGTHS=(256)
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

          accelerate launch src/run_ner.py \
          --run_name "$RUN_NAME" \
          --model_name_or_path "$MODEL_PATH/$MODEL" \
          --config_name "$MODEL_PATH/$MODEL" \
          --tokenizer_name "$MODEL_PATH/$MODEL" \
          --cache_dir "./cache/" \
          --logging_dir "./logs" \
          --output_dir "$OUTPUT_DIR" \
          --train_file "./data/ner/train.json" \
          --validation_file "./data/ner/dev.json" \
          --test_file "./data/ner/test.json" \
          --eval_strategy "steps" \
          --eval_steps 1000 \
          --per_device_train_batch_size $((batch/N_GPUS)) \
          --per_device_eval_batch_size $((batch/N_GPUS)) \
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
          --remove_unused_columns
        done
      done
    done
  done
done