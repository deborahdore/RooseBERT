#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false
python src/zs_argument_detection.py --model_id "mistralai/Mistral-7B-Instruct-v0.3"