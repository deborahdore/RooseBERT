#!/bin/bash

conda activate roosebert
export TOKENIZERS_PARALLELISM=false
python src/fs_sentiment_analysis.py --model_id "google/gemma-3-4b-it"