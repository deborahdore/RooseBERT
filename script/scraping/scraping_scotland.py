"""
Preprocessing Script for Scottish Parliament Debates
Download: https://dataverse.harvard.edu/file.xhtml?fileId=4432885&version=1.0
"""
import os

import pandas as pd
import rootutils
from tqdm import tqdm

from script.utils import clean_text, convert_to_dmy_format

# Set up project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define output path
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(), "data", "training")
os.makedirs(PATH_TO_FINAL_FOLDER, exist_ok=True)
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "scotland.csv")


def main(input_folder: str, output_file: str):
    """
    Load, clean, and merge Irish and UK plenary speeches into a single dataset.

    Args:
        input_folder (str): Path to the folder containing the raw CSV files.
        output_file (str): Path to save the cleaned combined CSV.
    """

    records = []
    file_path = os.path.join(input_folder, "parlScot_parl_v1.1.csv")

    print(f"✅ Opening dataset")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['is_speech'] == True]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        date = row['date']
        text = row['speech']
        speaker = row['name']
        date = convert_to_dmy_format(date, "%Y-%m-%d")
        records.append({
            "ID": f"ScotlandParliament_{date}",
            "date": date,
            "speaker": speaker,
            "text": text
        })

    df_combined = pd.DataFrame(records)
    df_combined["text"] = df_combined["text"].apply(clean_text)
    df_combined = df_combined.dropna().drop_duplicates().reset_index(drop=True)

    df_combined.to_csv(output_file, index=False)
    print("Dataset Length: ", len(df_combined))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    main(os.path.expanduser("~/Downloads/"), PATH_TO_FINAL_FILE)
    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
