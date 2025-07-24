"""
Computes the mean and standard deviation of F1 scores across multiple runs
for each model and task found in the `logs/` directory. Results are saved
to an Excel file (`random_seed_runs.xlsx`), with one sheet per task.
"""

import os
import statistics

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def extract_and_round(x):
    x_round = str(x).replace(",", ".")
    x_round = round(float(x_round), 3)
    return x_round


if __name__ == '__main__':
    # Each time a run_classification/run_ner is executed, it writes down the results in a csv file called random_seed_runs
    # After n runs, we can access it and extract mean and standard deviation
    with pd.ExcelWriter("logs/random_seed_runs.xlsx") as writer:
        for tsk in ['argument_detection', 'ner', 'relation_classification', 'sentiment_analysis']:
            models = []
            mean = []
            std = []
            scores = []

            path = os.path.join("logs", tsk)
            models_list = [model for model in os.listdir(path) if os.path.isdir(os.path.join(path, model))]
            for model in models_list:
                final_path = os.path.join(path, model)
                results = pd.read_csv(os.path.join(final_path, "random_seed_testing.csv"))
                assert (len(results) == 5)

                models.append(model)
                score = results['f1_score'].tolist()

                mean.append(extract_and_round(statistics.mean(score)))
                std.append(extract_and_round(statistics.stdev(score)))
                scores.append(score)

            df = pd.DataFrame({'Models': models, 'Mean': mean, 'Std': std, 'Scores': scores})
            df.to_excel(writer, sheet_name=tsk, index=False)
