#!/bin/bash
#SBATCH --job-name=perplexity
#SBATCH --gres=gpu:1

#SBATCH --output=logs/perplexity_%j.out
#SBATCH --error=logs/perplexity_%j.out


export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="Perplexity"
export HF_HOME="/home/ddore/.cache/huggingface/hub"

wandb disabled

# ------------------ HYPERPARAMETERS ------------------
MODELS=("FacebookAI/roberta-base" "allenai/longformer-base-4096" "answerdotai/ModernBERT-base" "ddore14/RooseBERT-cont-cased"
"ddore14/RooseBERT-cont-uncased" "ddore14/RooseBERT-scr-cased" "ddore14/RooseBERT-scr-uncased" "google-bert/bert-base-cased"
"google-bert/bert-base-uncased" "kornosk/polibertweet-political-twitter-roberta-mlm" "snowood1/ConfliBERT-cont-cased" "snowood1/ConfliBERT-cont-uncased"
"snowood1/ConfliBERT-scr-cased" "snowood1/ConfliBERT-scr-uncased" "microsoft/deberta-v3-base" "microsoft/deberta-base")

# FIRST TRAINING PHASE
MAX_STEPS_1=100000
MAX_SEQ_LEN_1=512
BATCH=8
GRAD_ACC=1
LR=1e-4

for model in "${MODELS[@]}"; do
  RUN_NAME="${model}-batch$((BATCH * N_GPUS * GRAD_ACC))-lr${LR}"
  printf "Starting training run: %s\n" "$RUN_NAME"

  mkdir -p "logs/${RUN_NAME}" "cache/${RUN_NAME}"

  # ------------------ TRAINING PHASE 1 ------------------

  python src/run_mlm.py \
          --model_name_or_path "$model" \
          --config_name "$model" \
          --cache_dir "cache/$RUN_NAME/" \
          --train_file "perplexity.csv" \
          --validation_file "perplexity.csv" \
          --max_seq_length "$MAX_SEQ_LEN_1" \
          --preprocessing_num_workers 8 \
          --output_dir "logs/$RUN_NAME/" \
          --do_eval \
          --eval_strategy "steps" \
          --per_device_train_batch_size $BATCH \
          --per_device_eval_batch_size $BATCH \
          --gradient_accumulation_steps $GRAD_ACC \
          --learning_rate $LR \
          --weight_decay 0.01 \
          --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
          --max_steps $MAX_STEPS_1 \
          --warmup_steps 10000 \
          --logging_dir "logs/$RUN_NAME/" \
          --logging_strategy "steps" \
          --logging_steps 500 \
          --save_strategy "steps" \
          --save_steps 20000 \
          --save_total_limit 1 \
          --seed 42 \
          --data_seed 42 \
          --fp16 \
          --local_rank 0 \
          --eval_steps 1000 \
          --dataloader_num_workers 8 \
          --run_name "$RUN_NAME" \
          --report_to "wandb" \
          --eval_on_start \
          --log_level "detail" \
          --overwrite_cache
done