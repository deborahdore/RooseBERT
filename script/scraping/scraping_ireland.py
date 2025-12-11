"""
Preprocessing Script for ParlEE Dataset
Download:
- https://dataverse.harvard.edu/file.xhtml?fileId=6435506&version=2.0
- https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6MZN76
"""

import os
from typing import List

import pandas as pd
import rootutils
from langdetect import detect, DetectorFactory, lang_detect_exception
from tqdm import tqdm

from script.utils import clean_text, convert_to_dmy_format

# Ensure consistent language detection
DetectorFactory.seed = 0

# Set up project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define output path
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(), "data", "training")
os.makedirs(PATH_TO_FINAL_FOLDER, exist_ok=True)
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "ireland.csv")


def is_english(text: str) -> bool:
    """Return True if text is detected as English, False otherwise."""
    try:
        return detect(text) == "en"
    except lang_detect_exception.LangDetectException:
        return False


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and missing values, reset index."""
    return df.dropna().drop_duplicates().reset_index(drop=True)


def parlee(input_folder: str) -> pd.DataFrame:
    """
    Load, clean, and merge Irish and UK plenary speeches into a single dataset.

    Args:
        input_folder (str): Path to the folder containing the raw CSV files.

    Returns:
        pd.DataFrame: Cleaned combined dataframe with columns ['date', 'text', 'speaker'].
    """
    records: List[dict] = []

    df = pd.read_csv(os.path.join(input_folder, "ParlEE_IE_plenary_speeches.csv"), low_memory=False)
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing rows", leave=False):
        date = convert_to_dmy_format(str(row.date), "%d/%m/%Y")
        text = clean_text(row.text)
        speaker = row.speaker
        if not text:  # Skip empty text
            continue
        records.append({"ID": "ParlEE_" + date, "date": date, "speaker": speaker, "text": text})

    df_combined = pd.DataFrame(records)
    return clean_dataframe(df_combined)


def ireland_parliament(file: str) -> pd.DataFrame:
    """
    Load Irish parliament debates and filter English speeches.

    Args:
        file (str): Path to the raw tab-delimited Irish parliament file.

    Returns:
        pd.DataFrame: Filtered dataframe with columns ['date', 'text', 'speaker'].
    """
    df = pd.read_csv(file, delimiter="\t")
    dataset: List[dict] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing Irish Parliament"):
        try:
            text = row.speech
            if not text or not is_english(text):
                continue
            dataset.append(
                {"ID": "IrishParliament_" + row.date,
                 "date": row.date,
                 "speaker": row.member_name,
                 "text": clean_text(text)
                 })
        except:
            continue
    return clean_dataframe(pd.DataFrame(dataset))


if __name__ == "__main__":
    parl_ee_path = os.path.expanduser("~/Downloads")
    irl_file = os.path.expanduser("~/Downloads/Dail_debates_1919-2013.tab")
    df_combined = pd.concat([parlee(parl_ee_path), ireland_parliament(irl_file)])
    df_combined = clean_dataframe(df_combined)

    df_combined.to_csv(PATH_TO_FINAL_FILE, index=False)
    print(f"Dataset Length: {len(df_combined)}")
    print(f"✅ Processed dataset saved to: {PATH_TO_FINAL_FILE}")

    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
