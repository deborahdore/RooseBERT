"""
Extracts evaluation results from model training logs in the `logs/` directory.

For each task and model, it reads metrics from `all_results.json` files,
parses run configurations from folder names, and compiles everything into
an Excel file (`results.xlsx`), with one sheet per task.
"""

import json
import os
import re

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# Define the regular expression pattern
pattern = r"(.*?)-?EPOCH(\d+)-LR([\d\.e\-]+)-WD([\d\.]+)-B(\d+)-ML(\d+)"

RESULTS_FOLDER = "logs/"

if __name__ == '__main__':
    tasks = [folder for folder in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, folder))]

    results_file_name = "results.xlsx"
    with pd.ExcelWriter(results_file_name) as writer:

        for task in tasks:
            task_path = os.path.join(RESULTS_FOLDER, task)
            models = [model for model in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, model))]
            data = []

            for model in models:
                model_path = os.path.join(task_path, model)
                runs = [run for run in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, run))]

                for run in runs:
                    run_path = os.path.join(model_path, run)

                    # Search for the pattern in the string
                    match = re.search(pattern, str(run))
                    epoch = int(match.group(2))
                    learning_rate = float(match.group(3))
                    weight_decay = float(match.group(4))
                    batch_size = int(match.group(5))
                    max_length = int(match.group(6))
                    type = None

                    with open(os.path.join(run_path, "all_results.json"), "r") as file:
                        all_results = json.load(file)

                    results = {
                        "run": run,
                        "model": model,
                        "type": model.split("-")[-1],
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "max_length": max_length,
                        "weight_decay": weight_decay,
                        "test_f1": all_results.get("test_f1", None),
                        "test_precision": all_results.get("test_precision", None),
                        "test_recall": all_results.get("test_recall", None),
                        "test_accuracy": all_results.get("test_accuracy", None),
                    }
                    data.append(results)

            if data:
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=task[:31], index=False)

    print(f"Results saved to '{results_file_name}'")
