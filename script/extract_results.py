"""
Extracts evaluation results from model training logs in the `logs/` directory.

For each task and model, it reads metrics from `all_results.json` files,
parses run configurations from folder names, and compiles everything into
Excel files:
- `results.xlsx`: all runs
- `best_results.xlsx`: best run per model and task
"""

import json
import os
import re
from pathlib import Path

import openpyxl
import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

RESULTS_FOLDER = Path("logs/encoders/")
RESULTS_FILE = Path("logs/encoders/") / "results.xlsx"
BEST_RESULTS_FILE = Path("logs/encoders/") / "best_results.xlsx"

RUN_PATTERN = re.compile(
    r"(.*?)-?EPOCH(?P<epoch>\d+)-LR(?P<lr>[\d.e\-]+)-WD(?P<wd>[\d.e\-]+)-B(?P<batch>\d+)"
)

METRICS = [
    "test_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
]

script_classification = """#!/bin/bash
#SBATCH --job-name={task}_{dataset}
#SBATCH --output=logs/{task}_{dataset}_%j.out
#SBATCH --error=logs/{task}_{dataset}_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --time=36:00:00

#SBATCH --account=marianne

set -e
module load miniconda
conda activate roosebert

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="{task}_{dataset}"
wandb disabled

export HF_HOME="/home/ddore/.cache/huggingface"

MODEL_DIR="{model_dir}"
SEEDS=()
wd=0.1

model="{model}"
lr={learning_rate}
batch={batch_size}
epoch={epoch}

for seed in "${{SEEDS[@]}}"; do

  RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
  OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"

  mkdir -p "$OUTPUT_DIR"

  python src/run_classification.py \\
    --run_name "$RUN_NAME" \\
    --model_name_or_path "$MODEL_DIR/$model" \\
    --config_name "$MODEL_DIR/$model" \\
    --tokenizer_name "$MODEL_DIR/$model" \\
    --cache_dir "./cache/" \\
    --logging_dir "./logs/" \\
    --output_dir "$OUTPUT_DIR" \\
    --train_file "./data/{task}/{dataset}/train.csv" \\
    --validation_file "./data/{task}/{dataset}/dev.csv" \\
    --test_file "./data/{task}/{dataset}/test.csv" \\
    --eval_strategy "steps" \\
    --eval_steps 2000 \\
    --logging_steps 1000 \\
    --save_total_limit 1 \\
    --per_device_train_batch_size "$batch" \\
    --per_device_eval_batch_size "$batch" \\
    --learning_rate "$lr" \\
    --weight_decay "$wd" \\
    --num_train_epochs "$epoch" \\
    --logging_strategy "steps" \\
    --save_strategy "epoch" \\
    --seed "$seed" \\
    --report_to "wandb" \\
    --text_column_name "text" \\
    --label_column_name "label" \\
    --eval_on_start \\
    --remove_unused_columns
done
    """
# todo
script_ner = """#!/bin/bash
#SBATCH --job-name=sequence_labelling_{dataset}
#SBATCH --output=logs/sequence_labelling_{dataset}_%j.out
#SBATCH --error=logs/sequence_labelling_{dataset}_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --time=36:00:00

#SBATCH --account=marianne

set -e
module load miniconda
conda activate roosebert

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="sequence_labelling_{dataset}"
wandb disabled

export HF_HOME="/home/ddore/.cache/huggingface"

MODEL_DIR="{model_dir}"
SEEDS=()
wd=0.1

model="{model}"
lr={learning_rate}
batch={batch_size}
epoch={epoch}

for seed in "${{SEEDS[@]}}"; do

    RUN_NAME=$(printf "%s-EPOCH%s-LR%s-WD%s-B%s" "$model" "$epoch" "$lr" "$wd" "$batch")
    OUTPUT_DIR="./logs/$WANDB_PROJECT/$model/$RUN_NAME"
    
    mkdir -p "$OUTPUT_DIR"
    
    python src/run_ner.py \\
    --run_name "$RUN_NAME" \\
    --model_name_or_path "$MODEL_DIR/$model" \\
    --config_name "$MODEL_DIR/$model" \\
    --tokenizer_name "$MODEL_DIR/$model" \\
    --cache_dir "./cache/" \\
    --logging_dir "./logs/" \\
    --output_dir "$OUTPUT_DIR" \\
    --train_file "./data/sequence_labelling/{dataset}/train.json" \\
    --validation_file "./data/sequence_labelling/{dataset}/dev.json" \\
    --test_file "./data/sequence_labelling/{dataset}/test.json" \\
    --eval_strategy "steps" \\
    --eval_steps 2000 \\
    --logging_steps 1000 \\
    --per_device_train_batch_size "$batch" \\
    --per_device_eval_batch_size "$batch" \\
    --learning_rate "$lr" \\
    --weight_decay "$wd" \\
    --num_train_epochs "$epoch" \\
    --logging_strategy "steps" \\
    --save_strategy "epoch" \\
    --save_total_limit 1 \\
    --seed "$seed" \\
    --report_to "wandb" \\
    --eval_on_start \\
    --remove_unused_columns
done
"""


def parse_run_name(run_name: str) -> dict | None:
    """Extract hyperparameters from a run directory name."""
    match = RUN_PATTERN.search(run_name)
    if not match:
        return None

    return {
        "epoch": int(match.group("epoch")),
        "learning_rate": float(match.group("lr")),
        "weight_decay": float(match.group("wd")),
        "batch_size": int(match.group("batch")),
    }


def load_results(run_path: Path) -> dict | None:
    """Load all_results.json if it exists."""
    results_file = run_path / "all_results.json"
    if not results_file.exists():
        return None

    with results_file.open() as f:
        return json.load(f)


def is_better(task: str, score: float, best_score: float) -> bool:
    """Decide whether a score is better depending on the task."""
    if task.startswith("binary"):
        return score > best_score
    return score > best_score


def extract_result() -> None:
    tasks = [p for p in RESULTS_FOLDER.iterdir() if p.is_dir()]

    all_results = {}
    best_results = {}

    for task_path in tasks:
        task = task_path.name
        all_rows = []
        best_rows = []

        for model_path in task_path.iterdir():
            if not model_path.is_dir():
                continue

            model = model_path.name
            best_score = float("-inf")
            best_row = None

            for run_path in model_path.iterdir():
                if not run_path.is_dir():
                    continue

                run_name = run_path.name
                params = parse_run_name(run_name)
                if params is None:
                    continue

                results = load_results(run_path)
                if results is None:
                    continue

                row = {
                    "task": task,
                    "model": model,
                    "type": model.split("-")[-1],
                    "run": run_name,
                    **params,
                }

                for metric in METRICS:
                    row[metric] = results.get(metric)

                # Select metric used for best model
                score = (
                    row["test_accuracy"]
                    if task.startswith("binary")
                    else row["test_f1"]
                )

                if score is not None and is_better(task, score, best_score):
                    best_score = score
                    best_row = row

                all_rows.append(row)

            if best_row is not None:
                best_rows.append(best_row)

        if all_rows:
            all_results[task] = pd.DataFrame(all_rows)
        if best_rows:
            best_results[task] = pd.DataFrame(best_rows)

    with pd.ExcelWriter(RESULTS_FILE) as writer:
        for task, df in all_results.items():
            sheet_name = (task.replace("binary_classification", "binary")
                          .replace("multi_class_classification", "multi_class")
                          .replace("sequence_labelling", "ner"))
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    with pd.ExcelWriter(BEST_RESULTS_FILE) as writer:
        for task, df in best_results.items():
            sheet_name = (task.replace("binary_classification", "binary")
                          .replace("multi_class_classification", "multi_class")
                          .replace("sequence_labelling", "ner"))
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Results saved to '{RESULTS_FILE}' and '{BEST_RESULTS_FILE}'")


# todo
MODELS2DIR = {
    'polibertweet-political-twitter-roberta-mlm': 'kornosk',
    'ConfliBERT-scr-uncased': 'snowood1',
    'ConfliBERT-scr-cased': 'snowood1',
    'ConfliBERT-cont-cased': 'snowood1',
    'ConfliBERT-cont-uncased': 'snowood1',
    'deberta-base': 'microsoft',
    'roberta-base': 'FacebookAI',
    'NEWRooseBERT-cont-uncased': 'ddore14',
    'NEWRooseBERT-cont-cased': 'ddore14',
    'NEWRooseBERT-scr-uncased': 'ddore14',
    'NEWRooseBERT-scr-cased': 'ddore14',
    'bert-base-uncased': 'google-bert',
    'bert-base-cased': 'google-bert'
}


def get_script_template(task_type: str) -> str:
    if task_type in {"binary_classification", "multi_class_classification"}:
        return script_classification
    elif task_type == "sequence_labelling":
        return script_ner
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def write_sbatch():
    tasks = openpyxl.load_workbook(BEST_RESULTS_FILE).sheetnames
    for t in tasks:
        os.makedirs(f"./sbatch/all/{t}", exist_ok=True)
        df = pd.read_excel(BEST_RESULTS_FILE, sheet_name=t)
        for _, row in df.iterrows():
            params = {
                "dataset": row.task.split("_")[-1],
                "task": row.task.replace(f"_{row.task.split('_')[-1]}", ""),
                "model": row.model,
                "model_dir": MODELS2DIR[row.model],
                "learning_rate": row.learning_rate,
                "batch_size": row.batch_size,
                "epoch": row.epoch,
            }
            template = get_script_template(params['task'])
            script = template.format(**params)
            # write to sh file
            with open(f"./sbatch/all/{t}/{t}_{params['model']}.sh", "w") as text_file:
                print(script, file=text_file)


if __name__ == "__main__":
    # extract_result()
    write_sbatch()
