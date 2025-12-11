"""
Australian Parliamentary Debates (1998-2022)

Download the hansard-corpus.zip file from:
https://zenodo.org/records/17351233/files/corpus_1998_to_2025.parquet?download=1
"""

import os
from pathlib import Path

import pandas as pd
import rootutils

from script.utils import clean_text, convert_to_dmy_format

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define paths
ROOT_DIR = rootutils.find_root()
PATH_TO_FINAL_FOLDER = Path(ROOT_DIR) / "data" / "training"
PATH_TO_FINAL_FOLDER.mkdir(parents=True, exist_ok=True)

# INPUT_FOLDER = Path.home() / "Downloads" / "hansard-corpus"
# INPUT_FILE = INPUT_FOLDER / "hansard_corpus_1998_to_2022.csv"
INPUT_FILE = Path.home() / "Downloads" / "corpus_1998_to_2025.parquet"
OUTPUT_FILE = PATH_TO_FINAL_FOLDER / "australia.csv"


def main():
    df = pd.read_parquet(INPUT_FILE)

    # Process columns
    df["date"] = df["date"].apply(convert_to_dmy_format)
    df["body"] = df["body"].apply(clean_text)
    df.rename(columns={'name': 'speaker', 'body': 'text'}, inplace=True)
    df['ID'] = "AustraliaHansard_" + df['date']
    df = df[['ID', 'date', 'speaker', 'text']]

    # Remove duplicates and missing values
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Save processed CSV
    df.to_csv(OUTPUT_FILE, index=False, escapechar='\\')

    print(f"Dataset Length: {len(df)}")
    print(f"✅ Processed dataset saved to: {OUTPUT_FILE}")

    os.system(f"du -sh {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
