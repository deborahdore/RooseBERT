"""
Preprocessing Script for South Africa and Ghana Parliament Debates
Download: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISX4G
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
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "africa.csv")


def main(input_folder: str, output_file: str):
    records = []
    files = [f for f in os.listdir(input_folder) if f.endswith("Speeches.csv")]

    print(f"✅ Opening dataset")
    for file in files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df)):
            date = row['Date']
            text = row['Speech']
            speaker = row['Speaker']
            date = convert_to_dmy_format(date, "%Y-%m-%d")
            parliament_type = "GhanaParliament_" if "Ghana" in file else "SouthAfrica_"
            records.append({
                "ID": f"{parliament_type}{date}",
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
    main(os.path.expanduser("~/Downloads/dataverse_files"), PATH_TO_FINAL_FILE)
    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
