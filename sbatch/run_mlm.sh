#!/bin/bash
#SBATCH --job-name=bert_base_uncased

#SBATCH -C a100
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

#SBATCH --time=20:00:00
#SBATCH --output=logs/bert_base_uncased_%j.out
#SBATCH --error=logs/bert_base_uncased_%j.out

#SBATCH --hint=nomultithread

module purge
module load arch/a100
module load cuda/12.4.1
module load miniforge/24.9.0

conda activate rooseBERT

export TOKENIZERS_PARALLELISM=false

wandb offline
export WANDB_MODE=offline
export WANDB_PROJECT="Masked_Language_Modelling"

export MASTER_PORT=6000
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ------------------ HYPERPARAMETERS ------------------
MODEL_NAME="bert-base-uncased"
MODEL_PATH="HuggingFace_Models/${MODEL_NAME}"
N_GPUS=8

# FIRST TRAINING PHASE
MAX_STEPS_1=120000
MAX_SEQ_LEN_1=128
BATCH=64
GRAD_ACC=4
LR=1e-4

# SECOND TRAINING PHASE
MAX_STEPS_2=150000
MAX_SEQ_LEN_2=512

RUN_NAME="${MODEL_NAME}-batch$((BATCH * N_GPUS * GRAD_ACC))-lr${LR}"
printf "Starting training run: %s\n" "$RUN_NAME"

mkdir -p "logs/${RUN_NAME}" "cache/${RUN_NAME}"

# ------------------ TRAINING PHASE 1 ------------------

python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
        --node_rank=${SLURM_PROCID} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d \
        src/run_mlm.py \
        --model_name_or_path "$MODEL_PATH" \
        --cache_dir "cache/$RUN_NAME/" \
        --train_file "data/training/max_128/train.csv" \
        --validation_file "data/training/max_128/dev.csv" \
        --max_seq_length "$MAX_SEQ_LEN_1" \
        --preprocessing_num_workers 4 \
        --output_dir "logs/$RUN_NAME/" \
        --do_train \
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
        --save_total_limit 3 \
        --seed 42 \
        --data_seed 42 \
        --fp16 \
        --local_rank 0 \
        --eval_steps 1000 \
        --dataloader_num_workers 8 \
        --run_name "$RUN_NAME" \
        --deepspeed "configs/deepspeed_config.json" \
        --report_to "wandb" \
        --eval_on_start \
        --log_level "detail" \
        --overwrite_cache

# ------------------ TRAINING PHASE 2 ------------------
CHECKPOINT_PATH="logs/${RUN_NAME}/checkpoint-120000"

python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
        --node_rank=${SLURM_PROCID} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --rdzv_backend=c10d \
        src/run_mlm.py \
        --model_name_or_path "$CHECKPOINT_PATH" \
        --overwrite_output_dir  \
        --resume_from_checkpoint "$CHECKPOINT_PATH" \
        --cache_dir "cache/$RUN_NAME/" \
        --train_file "data/training/max_512/train.csv" \
        --validation_file "data/training/max_512/dev.csv" \
        --max_seq_length "$MAX_SEQ_LEN_2" \
        --preprocessing_num_workers 4 \
        --output_dir "logs/$RUN_NAME/" \
        --do_train \
        --do_eval \
        --eval_strategy "steps" \
        --per_device_train_batch_size $BATCH \
        --per_device_eval_batch_size $BATCH \
        --gradient_accumulation_steps $GRAD_ACC \
        --learning_rate $LR \
        --weight_decay 0.01 \
        --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
        --max_steps $MAX_STEPS_2 \
        --logging_dir "logs/$RUN_NAME/" \
        --logging_strategy "steps" \
        --logging_steps 500 \
        --save_strategy "steps" \
        --save_steps 20000 \
        --save_total_limit 3 \
        --seed 42 \
        --data_seed 42 \
        --fp16 \
        --local_rank 0 \
        --eval_steps 1000 \
        --dataloader_num_workers 8 \
        --run_name "$RUN_NAME" \
        --deepspeed "configs/deepspeed_config.json" \
        --report_to "wandb" \
        --eval_on_start \
        --log_level "detail" \
        --overwrite_cache