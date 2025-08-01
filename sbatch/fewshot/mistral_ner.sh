#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false
python src/fs_ner.py --model_id "mistralai/Mistral-7B-Instruct-v0.3"