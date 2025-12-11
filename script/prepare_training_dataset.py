"""
Script used to prepare data for the masked language modelling task.
"""

import logging
import os
import random
from collections import Counter
from itertools import combinations
from typing import List, Tuple

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
DATA_DIR = os.path.join(BASE_PATH, 'data/training')


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

def load_and_process_data(max_sequences: List[int] = [64, 128, 256, 512]) -> Tuple:
    """Load CSVs, split sentences, create chunks, split into train/dev."""
    all_train, all_dev = [], []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith(("train", "dev"))]
    logger.info("Found %d CSV files in %s", len(files), DATA_DIR)

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(DATA_DIR, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read %s: %s", file, e)
            raise
        if 'text' not in df.columns:
            logger.warning("Skipping %s: No 'text' column found", file)
            continue

        df = split_sentences(df)

        train, dev = train_test_split(df, test_size=0.1, random_state=42, shuffle=False)
        all_train.append(train)
        all_dev.append(dev)

    if not all_train or not all_dev:
        raise ValueError("No data processed.")

    train_df = pd.concat(all_train).reset_index(drop=True)
    dev_df = pd.concat(all_dev).reset_index(drop=True)

    train_chunks = {k: concatenate_in_chunks(train_df, k) for k in max_sequences}
    dev_chunks = {k: concatenate_in_chunks(dev_df, k) for k in max_sequences}

    # Shuffle datasets
    train_chunks = {k: shuffle_df(v) for k, v in train_chunks.items()}
    dev_chunks = {k: shuffle_df(v) for k, v in dev_chunks.items()}

    return train_chunks, dev_chunks


def main():
    for size in [128, 512]:
        train_chunks, dev_chunks = load_and_process_data(max_sequences=[size // 2, size])
        os.makedirs(os.path.join(DATA_DIR, f"max_{size}"), exist_ok=True)
        train = train_chunks[size]
        dev = dev_chunks[size]

        # Same speaker / same debate
        train['same_speaker'] = True
        dev['same_speaker'] = True
        train['same_debate'] = True
        dev['same_debate'] = True

        # Different speaker same debate
        train_diff_speaker = create_speaker_change_prediction_dataset(train_chunks[size // 2], len(train) // 2)
        dev_diff_speaker = create_speaker_change_prediction_dataset(dev_chunks[size // 2], len(dev) // 2)
        train_diff_speaker['same_speaker'] = False
        dev_diff_speaker['same_speaker'] = False
        train_diff_speaker['same_debate'] = True
        dev_diff_speaker['same_debate'] = True

        # Different debate
        train_diff_debate = create_argument_continuity_dataset(train_chunks[size // 2], len(train) // 2)
        dev_diff_debate = create_argument_continuity_dataset(dev_chunks[size // 2], len(dev) // 2)
        train_diff_debate['same_speaker'] = False
        dev_diff_debate['same_speaker'] = False
        train_diff_debate['same_debate'] = False
        dev_diff_debate['same_debate'] = False

        # Concatenate and save
        if size == 512:
            dev.to_csv(os.path.join(DATA_DIR, "perplexity_test.csv"), index=False)

        train_full = pd.concat([train, train_diff_speaker, train_diff_debate], axis=0)
        train_full = clean_df(train_full, 'text')

        dev_full = pd.concat([dev, dev_diff_speaker, dev_diff_debate], axis=0)
        dev_full = clean_df(dev_full, 'text')

        train_full.to_csv(os.path.join(DATA_DIR, f'max_{size}/train.csv'), index=False)
        dev_full.to_csv(os.path.join(DATA_DIR, f'max_{size}/dev.csv'), index=False)

        logger.info("Saved datasets for max_sequence=%d", size)
        logger.info("Train size: %d | Dev size: %d", len(train_full), len(dev_full))


if __name__ == '__main__':
    main()
