"""
Preprocessing Script for ParlEE Dataset

This script consolidates debates from the UN Security Council into a single cleaned CSV file.

Instructions:
1. Download the following files from Harvard Dataverse:
   - https://dataverse.harvard.edu/file.xhtml?fileId=10809805&version=6.1
2. Provide the folder path containing these files using the --path_to_downloaded_folder argument.

Example usage:
    python preprocess_UNSC.py --path_to_downloaded_folder /path/to/downloads
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "unsc.csv")


def preprocess_unsc(input_folder, final_file):
    files = os.listdir(input_folder)
    dataset = []
    for file in tqdm(files):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(input_folder, file)

        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            continue

        date = convert_to_dmy_format(str(file.split("_")[1]), "%Y")
        dataset.append({
            'text': clean_text(content),
            'date': date
        })
    df = pd.DataFrame(dataset)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file, index=False)

    print("Dataset Length: {}".format(len(df)))
    print(f"✅ Processed dataset saved to: {final_file}")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Preprocess UN Security Council Data")
    # parser.add_argument("--path_to_downloaded_folder", type=str, required=True,
    #                     help="Path to the folder containing the unzipped UNGDC session folders")
    #
    # args = parser.parse_args()
    #
    # print(f"📂 Loading data from: {args.path_to_downloaded_folder}")
    # preprocess_unsc(args.path_to_downloaded_folder, PATH_TO_FINAL_FILE)

    # Quick local testing without CLI
    preprocess_unsc("/Downloads/speeches", PATH_TO_FINAL_FILE)

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
