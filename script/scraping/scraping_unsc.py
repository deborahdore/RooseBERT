"""
Preprocessing Script for ParlEE Dataset
Download: https://dataverse.harvard.edu/file.xhtml?fileId=10809805&version=6.1
"""
import os
import re

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


def main(input_folder, final_file):
    files = os.listdir(input_folder)
    dataset = []
    idx = 0
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
        cleaned = clean_text(content)
        match = re.match(r'^([^(]+)', cleaned)
        if match:
            speaker = match.group(1).strip()
        else:
            speaker = "Speaker" + str(idx)
            idx += 1
        dataset.append({
            'ID': f"UNSC_{date}",
            'date': date,
            'speaker': speaker,
            'text': clean_text(content),
        })
    df = pd.DataFrame(dataset)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_csv(final_file, index=False)

    print("Dataset Length: {}".format(len(df)))
    print(f"✅ Processed dataset saved to: {final_file}")


if __name__ == '__main__':
    main(os.path.expanduser("~/Downloads/speeches"), PATH_TO_FINAL_FILE)
    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
