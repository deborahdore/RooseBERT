#!/bin/bash

conda activate roosebert
export TOKENIZERS_PARALLELISM=false
python src/fs_argument_detection.py --model_id "meta-llama/Llama-3.1-8B-Instruct"