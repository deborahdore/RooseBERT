"""
Preprocessing Script for ParlEE Dataset

This script consolidates debates from the Irish and UK Parliaments into a single cleaned CSV file.

Instructions:
1. Download the following files from Harvard Dataverse:
   - https://dataverse.harvard.edu/file.xhtml?fileId=6435506&version=2.0 (IE debates)
   - https://dataverse.harvard.edu/file.xhtml?fileId=6435505&version=2.0 (UK debates)

2. Provide the folder path containing these files using the --path_to_downloaded_folder argument.

Example usage:
    python preprocess_parlee.py --path_to_downloaded_folder /path/to/downloads
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "ParlEE.csv")


def preprocess_files(input_folder: str, output_file: str):
    """
    Load, clean, and merge Irish and UK plenary speeches into a single dataset.

    Args:
        input_folder (str): Path to the folder containing the raw CSV files.
        output_file (str): Path to save the cleaned combined CSV.
    """
    file_paths = [
        os.path.join(input_folder, "ParlEE_IE_plenary_speeches.csv"),
        os.path.join(input_folder, "ParlEE_UK_plenary_speeches.csv")
    ]

    records = []

    for file_path in tqdm(file_paths, desc="Processing files"):
        df = pd.read_csv(file_path, low_memory=False)
        for date, text in zip(df["date"], df["text"]):
            date = convert_to_dmy_format(date_str=date, current_format="%d/%m/%Y")
            records.append({"date": date, "text": text})

    df_combined = pd.DataFrame(records).dropna().drop_duplicates().reset_index(drop=True)
    df_combined["text"] = df_combined["text"].apply(clean_text)

    df_combined.to_csv(output_file, index=False)
    print("Dataset Length: {}".format(len(df_combined)))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess UK and Irish Parliament debates")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing downloaded debate files")
    #
    # args = parser.parse_args()
    # preprocess_files(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_files("/Downloads", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
