"""
Script used to prepare data for the masked language modelling task.
"""
import argparse
import logging
import os
import random
from collections import Counter
from itertools import combinations
from typing import Tuple

import nltk
import pandas as pd
import rootutils
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------- Setup -------------------

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Setup project root and data directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
BASE_PATH = rootutils.find_root(search_from=__file__, indicator=".project-root")


# ------------------- Helpers -------------------

def split_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Split text into individual sentences."""
    logger.debug("Splitting text into sentences...")
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).apply(sent_tokenize)
    return df.explode("text").reset_index(drop=True)


def concatenate_in_chunks(df: pd.DataFrame, max_sequence: int) -> pd.DataFrame:
    """Concatenate sentences into chunks not exceeding max_sequence words."""
    logger.debug("Concatenating sentences into chunks of max %d words", max_sequence)
    results = []

    group_cols = [c for c in ["ID", "date", "speaker"] if c in df.columns]

    for group_values, group_df in df.groupby(group_cols):
        sentences = group_df["text"].fillna("").tolist()
        current_chunk, current_len = [], 0
        chunks = []

        for sent in sentences:
            words = sent.split()
            if current_len + len(words) <= max_sequence:
                current_chunk.append(sent)
                current_len += len(words)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        for chunk in chunks:
            row = {col: val for col, val in zip(group_cols, group_values)}
            row["text"] = chunk
            results.append(row)

    logger.debug("Total chunks created: %d", len(results))
    return pd.DataFrame(results)


def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle and remove duplicates."""
    return df.sample(frac=1).dropna().drop_duplicates().reset_index(drop=True)


# ------------------- Dataset Generators -------------------

def count_diff_speaker_pairs(rows):
    # Count how many rows each speaker has
    speaker_counts = Counter(row['speaker'] for row in rows)

    # Sum the product of counts for each pair of different speakers
    total_pairs = sum(speaker_counts[s1] * speaker_counts[s2]
                      for s1, s2 in combinations(speaker_counts.keys(), 2))

    return total_pairs


def create_speaker_change_prediction_dataset(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Create examples combining sentences from different speakers on the same debate.
        Speaker Change Prediction: Predict whether the next utterance comes from the same or different speaker. ù
        Captures turn-taking dynamics
    """
    dataset = []
    num_debates = len(df['ID'].unique())
    for ID, debate_df in df.groupby("ID"):
        for date, date_df in debate_df.groupby("date"):
            rows = date_df.to_dict("records")
            speakers = date_df["speaker"].unique()
            if len(speakers) < 2:
                num_debates = num_debates - 1
                continue
            count = 0
            new_target = min(count_diff_speaker_pairs(rows), target // num_debates)
            while count < new_target:
                r1, r2 = random.sample(rows, 2)
                if r1["speaker"] == r2["speaker"]:
                    continue
                dataset.append({
                    "ID": ID,
                    "date": date,
                    "speaker": f"{r1['speaker']}_{r2['speaker']}",
                    "text": f"{r1['text']} {r2['text']}",
                })
                count += 1
    return pd.DataFrame(dataset).dropna().drop_duplicates().reset_index(drop=True)


def create_argument_continuity_dataset(df: pd.DataFrame, target: int) -> pd.DataFrame:
    """Create examples combining sentences from different debates.
        Argument Continuity Modeling: Predict if consecutive sentences belong to the same argumentative thread
    """
    dataset = []
    rows = df.to_dict("records")
    while len(dataset) < target:
        r1, r2 = random.sample(rows, 2)
        if r1["ID"] == r2["ID"] or r1["ID"].split("_")[0] == r2["ID"].split("_")[0]:
            continue
        dataset.append({
            "ID": f"{r1['ID']} {r2['ID']}",
            "date": f"{r1['date']} {r2['date']}",
            "speaker": f"{r1['speaker']}_{r2['speaker']}",
            "text": f"{r1['text']} {r2['text']}",
        })
    return pd.DataFrame(dataset).dropna().drop_duplicates().reset_index(drop=True)


def clean_df(df: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
    df = df.dropna().drop_duplicates()
    df = df[df[text_column_name].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    return df.reset_index(drop=True)


# ------------------- Main Workflow -------------------

def load_and_process_data(data_dir, max_sequence: int) -> Tuple:
    """Load CSVs, split sentences, create chunks, split into train/dev."""
    all_train, all_dev = [], []

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and not f.startswith(("train", "dev"))]
    logger.info("Found %d CSV files in %s", len(files), data_dir)

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read %s: %s", file, e)
            raise
        if 'text' not in df.columns:
            logger.warning("Skipping %s: No 'text' column found", file)
            continue

        df = split_sentences(df)
        df = concatenate_in_chunks(df, max_sequence)

        train, dev = train_test_split(df, test_size=0.1, random_state=42, shuffle=False)
        all_train.append(train)
        all_dev.append(dev)

    if not all_train or not all_dev:
        raise ValueError("No data processed.")

    return pd.concat(all_train).reset_index(drop=True), pd.concat(all_dev).reset_index(drop=True)


def main(data_dir: str):
    for size in [128, 512]:
        train, dev = load_and_process_data(data_dir, max_sequence=size)
        os.makedirs(os.path.join(data_dir, f"max_{size}"), exist_ok=True)

        if size == 512:
            dev.to_csv(os.path.join(data_dir, "perplexity_test.csv"), index=False)

        train = clean_df(train, 'text')
        dev = clean_df(dev, 'text')

        train.to_csv(os.path.join(data_dir, f'max_{size}/train.csv'), index=False)
        dev.to_csv(os.path.join(data_dir, f'max_{size}/dev.csv'), index=False)

        logger.info("Saved datasets for max_sequence=%d", size)
        logger.info("Train size: %d | Dev size: %d", len(train), len(dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="training")
    args = parser.parse_args()

    main(os.path.join(BASE_PATH, f'{rootutils.find_root(__file__)}/data/{args.dataset}'))
