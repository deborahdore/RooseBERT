"""
Preprocessing Script for Scottish Parliament Debates

This script processes text files from the Scottish Parliament dataset,
cleans the content, and consolidates them into a single CSV file.

Instructions:
1. Download the file from: https://dataverse.harvard.edu/file.xhtml?fileId=4432885&version=1.0
2. Unzip the archive and pass the extracted folder path using --path_to_downloaded_folder

Example usage:
    python preprocess_scots_parl.py --path_to_downloaded_folder /path/to/parlScot_parl_v1.1.csv
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "scottish_parliament.csv")


def preprocess_scots_parliament(input_folder: str, output_file: str):
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

    for date, text in tqdm(zip(df["date"], df["speech"]), total=len(df)):
        date = convert_to_dmy_format(date, "%Y-%m-%d")
        records.append({"date": date, "text": text})

    df_combined = pd.DataFrame(records).dropna().drop_duplicates().reset_index(drop=True)
    df_combined["text"] = df_combined["text"].apply(clean_text)

    df_combined.to_csv(output_file, index=False)
    print("Dataset Length: ", len(df_combined))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess Scottish Parliament debates")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing unzipped text files")
    #
    # args = parser.parse_args()
    #
    # preprocess_scots_parliament(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_scots_parliament("/Downloads/", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
