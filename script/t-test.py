import ast
import json

import openpyxl
import pandas as pd
import rootutils
from scipy.stats import ttest_rel

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def ttest(my_model, comparison_model):
    t_stat, p_value = ttest_rel(my_model, comparison_model)
    # Interpretation
    # if p_value < 0.05:
    #     print("✅ Statistically significant improvement (p < 0.05)")
    # else:
    #     print("⚠️ No statistically significant difference (p ≥ 0.05)")

    # print(f"T-statistic: {t_stat:.5f}")
    # print(f"P-value: {p_value:.5f}")
    return p_value


if __name__ == '__main__':
    file_path = "./logs/encoders/5runs/random_seed_runs.xlsx"
    tasks = openpyxl.load_workbook(file_path).sheetnames
    for t in tasks:
        statistically_significant = {}
        df = pd.read_excel(file_path, sheet_name=t)
        df['Scores'] = df['Scores'].apply(ast.literal_eval)
        best_model = df.iloc[df['Mean'].idxmax()]
        print(f"Best model: {best_model['Models']} with task {t}")
        df.sort_values(by="Models", ascending=True, inplace=True)
        for _, row in df.iterrows():
            if row['Models'] == best_model['Models']: continue
            p_value = ttest(my_model=best_model['Scores'], comparison_model=row['Scores'])
            if p_value < 0.01:
                statistically_significant[row['Models']] = {
                    "p_value": "p<0.01",
                    "value": "*" * 3,
                    "mean": row['Mean'],
                    "std": row['Std']
                }
            elif p_value < 0.05:
                statistically_significant[row['Models']] = {
                    "p_value": "p<0.05",
                    "value": "*" * 2,
                    "mean": row['Mean'],
                    "std": row['Std']
                }
            elif p_value < 0.1:
                statistically_significant[row['Models']] = {
                    "p_value": "p<0.1",
                    "value": "*" * 1,
                    "mean": row['Mean'],
                    "std": row['Std']
                }
            else:
                statistically_significant[row['Models']] = {
                    "p_value": p_value,
                    "value": "not significant",
                    "mean": row['Mean'],
                    "std": row['Std']
                }
        print(json.dumps(statistically_significant, sort_keys=False, indent=4))
