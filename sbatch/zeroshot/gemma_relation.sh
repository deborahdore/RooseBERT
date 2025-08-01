#!/bin/bash
conda activate roosebert
export TOKENIZERS_PARALLELISM=false

python src/zs_relation_classification.py --model_id "google/gemma-3-4b-it"