#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home/user/.cache/huggingface/hub"

python src/fs_argument_detection.py --model_id "google/gemma-3-4b-it"
python src/fs_relation_classification.py --model_id "google/gemma-3-4b-it"
python src/fs_sentiment_analysis.py --model_id "google/gemma-3-4b-it"
python src/fs_stance_detection.py --model_id "google/gemma-3-4b-it"

python src/zs_argument_detection.py --model_id "google/gemma-3-4b-it"
python src/zs_relation_classification.py --model_id "google/gemma-3-4b-it"
python src/zs_sentiment_analysis.py --model_id "google/gemma-3-4b-it"
python src/zs_stance_detection.py --model_id "google/gemma-3-4b-it"