#!/bin/bash
#SBATCH --job-name=mistral

#SBATCH -C a100
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2

#SBATCH --time=20:00:00
#SBATCH --output=logs/mistral_argument_detection_%j.out
#SBATCH --error=logs/mistral_argument_detection_%j.out

#SBATCH --hint=nomultithread

module purge
module load arch/a100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate roosebert

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="Large-Language-Models_argument_detection"
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
            --train_file "./data/argument_detection/train.json" \
            --validation_file "./data/argument_detection/dev.json" \
            --test_file "./data/argument_detection/test.json" \
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
            --report_to "wandb"
        done
      done
    done
  done
done