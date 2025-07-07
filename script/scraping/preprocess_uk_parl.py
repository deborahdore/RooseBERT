"""
Preprocessing Script for UK Parliament Debates

This script consolidates CSV files containing UK Parliament debates into a single cleaned CSV file.

Instructions:
1. Download the corpus from:
   https://reshare.ukdataservice.ac.uk/854292/

2. Unzip it and provide the path to the extracted folder using --path_to_downloaded_folder

Example usage:
    python preprocess_uk_parl.py --path_to_downloaded_folder /path/to/unzipped/out
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "uk_parliament.csv")


def preprocess_uk_parliament(input_folder: str, output_file: str):
    """
    Loads and consolidates UK Parliament CSV files into a single cleaned CSV file.

    Args:
        input_folder (str): Path to the folder containing the debate CSV files.
        output_file (str): Path to save the final cleaned dataset.
    """
    all_records = []
    files = sorted(
        f for f in os.listdir(input_folder)
        if f.endswith(".csv") and os.path.isfile(os.path.join(input_folder, f))
    )

    print(f"📂 Found {len(files)} CSV files to process...")

    for filename in tqdm(files, desc="Processing files"):
        date = os.path.splitext(filename)[0]
        file_path = os.path.join(input_folder, filename)

        try:
            df = pd.read_csv(file_path, usecols=["body"], low_memory=False)
            df["date"] = convert_to_dmy_format(date, "%Y")
            all_records.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {filename} due to error: {e}")

    if all_records:
        df_combined = pd.concat(all_records, ignore_index=True)
        df_combined.rename(columns={"body": "text"}, inplace=True)
        df_combined = df_combined.dropna().drop_duplicates().reset_index(drop=True)
        df_combined["text"] = df_combined["text"].apply(clean_text)
        df_combined.to_csv(output_file, index=False)
        print("Dataset Length: {}".format(len(df_combined)))
        print(f"✅ Processed dataset saved to: {output_file}")
    else:
        print("❌ No valid files found. Nothing to save.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess UK Parliament debates")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing downloaded CSV files")
    #
    # args = parser.parse_args()
    # print(f"🔍 Loading data from: {args.path_to_downloaded_folder}")
    # preprocess_uk_parliament(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_uk_parliament("/Downloads/out", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
