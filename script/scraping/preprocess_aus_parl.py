"""
Preprocessing Script for Australian Parliament Debates

This script processes Hansard text files from the Australian Parliament dataset,
cleans the content, and consolidates them into a single CSV file.

Instructions:
1. Download the hansard-daily-csv.zip file from:
   https://zenodo.org/records/8121950

2. Unzip the archive and pass the extracted folder path using --path_to_downloaded_folder

Example usage:
    python preprocess_aus_parl.py --path_to_downloaded_folder /path/to/hansard-daily-csv
"""
import os

import pandas as pd
import rootutils
from tqdm import tqdm

from script.utils import clean_text, load_file_body, convert_to_dmy_format

# Set up project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define output path
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(), "data", "training")
os.makedirs(PATH_TO_FINAL_FOLDER, exist_ok=True)
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "australian_parliament.csv")


def preprocess_aus_parliament(input_folder: str, output_file: str):
    """
    Loads, cleans, and consolidates debate text files from the Australian Parliament.

    Args:
        input_folder (str): Path to the folder containing raw text files.
        output_file (str): Path to save the cleaned combined CSV.
    """
    dataset = []

    files = sorted(
        f for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    )

    print(f"📂 Found {len(files)} text files to process...")

    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_folder, filename)
        full_text = load_file_body(file_path)

        date = convert_to_dmy_format(date_str=filename.split(".")[0], current_format="%Y-%m-%d")
        dataset.append({
            "date": date,
            "text": full_text
        })

    df = pd.DataFrame(dataset)
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    df.to_csv(output_file, index=False, escapechar='\\')
    print("Dataset Length:", len(df))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess Australian Parliament debates")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing unzipped text files")
    #
    # args = parser.parse_args()
    #
    # preprocess_aus_parliament(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_aus_parliament("/Downloads/hansard-daily-csv", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
