"""
Extracts evaluation results from model training logs in the `logs/` directory.

For each task and model, it reads metrics from `all_results.json` files,
parses run configurations from folder names, and compiles everything into
Excel files:
- `results.xlsx`: all runs
- `best_results.xlsx`: best run per model and task
"""

import json
import re
from pathlib import Path

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

RESULTS_FOLDER = Path("logs")
RESULTS_FILE = Path("logs") / "results.xlsx"
BEST_RESULTS_FILE = Path("logs") / "best_results.xlsx"

RUN_PATTERN = re.compile(
    r"(.*?)-?EPOCH(?P<epoch>\d+)-LR(?P<lr>[\d.e\-]+)-WD(?P<wd>[\d.e\-]+)-B(?P<batch>\d+)"
)

METRICS = [
    "test_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
]


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


def main() -> None:
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
            df.to_excel(writer, sheet_name=task[:31], index=False)

    with pd.ExcelWriter(BEST_RESULTS_FILE) as writer:
        for task, df in best_results.items():
            df.to_excel(writer, sheet_name=task[:31], index=False)

    print(f"Results saved to '{RESULTS_FILE}' and '{BEST_RESULTS_FILE}'")


if __name__ == "__main__":
    main()
