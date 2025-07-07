"""
Preprocessing Script for European Parliament Debates

This script processes daily CSV files from the European Parliament Hansard dataset,
filters for English-language entries, and consolidates them into a single CSV file.

Instructions:
1. Download the ZIP archive from:
   https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/XPCVEI/ORRNK4&version=3.0

2. Unzip the archive and pass the extracted folder path using --path_to_downloaded_folder

Example usage:
    python preprocess_europarl.py --path_to_downloaded_folder /path/to/cleaned_speeches
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "european_parliament.csv")


def preprocess_europarl(input_folder: str, output_file: str):
    """
    Load, clean, and merge all English-language debate transcripts from the European Parliament.

    Args:
        input_folder (str): Path to the folder containing the daily CSV files.
        output_file (str): Path to save the cleaned combined CSV.
    """
    dataset = []

    files = sorted(
        f for f in os.listdir(input_folder)
        if f.endswith(".csv") and os.path.isfile(os.path.join(input_folder, f))
    )

    print(f"📂 Found {len(files)} CSV files to process...")

    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Rename columns to a consistent schema
        df.columns = ["argument", "date", "place", "speaker", "n", "text", "cleaned_text", "lang"]

        # Keep only English-language speeches
        df = df[df["lang"] == "en"]

        for date, text in zip(df["date"], df["text"]):
            date = convert_to_dmy_format(date_str=date, current_format="%d-%m-%Y")
            dataset.append({"date": date, "text": text})

    df_final = pd.DataFrame(dataset)
    df_final["text"] = df_final["text"].apply(clean_text)
    df_final = df_final.dropna().drop_duplicates().reset_index(drop=True)

    df_final.to_csv(output_file, index=False)
    print("Dataset Length: {}".format(len(df_final)))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess European Parliament debates")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing unzipped CSV files")
    #
    # args = parser.parse_args()
    # preprocess_europarl(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_europarl("/Downloads/Users/hjms/Desktop/Cleaned_Speeches", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
