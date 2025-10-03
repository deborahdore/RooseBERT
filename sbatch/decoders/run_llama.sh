#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --output=logs/llama.out

#SBATCH --time=36:00:00
#SBATCH --account=marianne
#SBATCH --gpus=1
#SBATCH --partition=gpu

conda activate roosebert
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/ddore/.cache/huggingface/hub"

python src/fs_argument_detection.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/fs_relation_classification.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/fs_sentiment_analysis.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/fs_stance_detection.py --model_id "meta-llama/Llama-3.1-8B-Instruct"

python src/zs_argument_detection.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/zs_relation_classification.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/zs_sentiment_analysis.py --model_id "meta-llama/Llama-3.1-8B-Instruct"
python src/zs_stance_detection.py --model_id "meta-llama/Llama-3.1-8B-Instruct"