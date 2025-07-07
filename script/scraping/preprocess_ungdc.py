"""
Preprocessing Script for the United Nations General Debate Corpus (UNGDC)

This script consolidates text files from the UNGDC into a single CSV file for downstream use.

Instructions:
1. Download the corpus from:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y

2. Unzip the archive and provide the extracted folder using --path_to_downloaded_folder

Example usage:
    python preprocess_ungdc.py --path_to_downloaded_folder /path/to/TXT
"""

import os

import pandas as pd
import rootutils
from tqdm import tqdm

from script.utils import clean_text, convert_to_dmy_format

# Set up project root and define output paths
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(), "data", "training")
os.makedirs(PATH_TO_FINAL_FOLDER, exist_ok=True)
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "ungdc.csv")

# Constants
START_YEAR = 1946
END_YEAR = 2024
SESSIONS = [f"{n:02d}" for n in range(1, 79)]  # Format session numbers as two digits


def preprocess_ungdc(input_folder: str, output_file: str):
    """
    Processes UNGDC session files into a cleaned DataFrame and saves to CSV.

    Args:
        input_folder (str): Root folder containing UNGDC session folders.
        output_file (str): Destination path for the consolidated CSV.
    """
    records = []

    # Loop over sessions from 1946 to 2023 (session 01 to 78)
    for session_num, year in tqdm(zip(SESSIONS, range(START_YEAR, END_YEAR)),
                                  total=END_YEAR - START_YEAR,
                                  desc="Processing sessions"):
        session_dir = os.path.join(input_folder, f"Session {session_num} - {year}")

        # Skip sessions that are missing
        if not os.path.exists(session_dir):
            print(f"⚠️ Skipping missing folder: {session_dir}")
            continue

        # Iterate over text files in the session folder
        for filename in os.listdir(session_dir):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(session_dir, filename)

            try:
                with open(file_path, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue

            cleaned = clean_text(content)

            records.append({
                "text": cleaned,
                "title": f"Session {session_num} - {year}",
                "date": convert_to_dmy_format(str(year), "%Y")
            })

    # Create and save DataFrame
    df = pd.DataFrame(records)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)
    df.to_csv(output_file, index=False, escapechar='\\')
    print("Dataset Length: {}".format(len(df)))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Preprocess UNGDC (UN General Debate Corpus)")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing the unzipped UNGDC session folders")
    #
    # args = parser.parse_args()
    #
    # print(f"📂 Loading data from: {args.path_to_downloaded_folder}")
    # preprocess_ungdc(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_ungdc("/Downloads/TXT", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
