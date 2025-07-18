"""
Run the download.sh script before this.
This file will preprocess all the datasets needed for the downstream tasks.
"""

import json
import logging
import os
import re

import pandas as pd
import rootutils
from sklearn.model_selection import train_test_split
from unidecode import unidecode

# Set up the root directory so relative paths work across the project
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Configure logging for tracking progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def adjust_conll_data(df: pd.DataFrame):
    """
    Adjust and clean CONLL data to ensure the token and NER tag lengths match.
    This handles edge cases where the tag list may be longer than the token list.
    """
    data = {"tokens": [], "ner_tags": []}
    for token, tag in zip(df['tokens'], df['ner_tags']):
        tag = tag[len(tag) - len(token):] if len(token) < len(tag) else tag
        data["tokens"].append(token)
        data["ner_tags"].append(tag)
    df = pd.DataFrame(data).dropna().reset_index(drop=True)
    return df


def save_conll_data(df: pd.DataFrame, file_path: str):
    """
    Save tokenized CONLL-style data as a JSON file.
    Each sentence is stored with an ID, tokens, and corresponding NER tags.
    """
    json_data = []
    for i, row in df.iterrows():
        json_obj = {
            "id": str(i),
            "ner_tags": row['ner_tags'],
            "tokens": row['tokens']
        }
        json_data.append(json_obj)

    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    logging.info(f"Saved CONLL data to {file_path}")


def matches_uppercase_name_colon(phrase: str) -> bool:
    pattern = r'(?m)^(?:[A-Z]+(?:\s[A-Z]+)*)\s*:'
    return re.match(pattern, phrase.strip()) is not None


def preprocess_argument_detection(folder):
    """
    Preprocess argument detection data from CONLL-style text files.
    Each line contains a token and its tag, and sentences are separated by blank lines.
    Data is split into train/dev/test sets and saved in JSON format.
    """
    files = os.listdir(folder)
    file_map = {
        'disputool-validation.conll': 'dev.json',
        'disputool-test.conll': 'test.json',
        'disputool-train.conll': 'train.json'
    }
    for f in files:
        data = []
        file_path = os.path.join(folder, f)
        current_sentence = {"tokens": [], "ner_tags": []}

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Group by speech turn
                line = line.strip()
                if line:
                    token, tag = line.split('\t')[1], line.split('\t')[-1]
                    token = unidecode(token)
                    if matches_uppercase_name_colon(token):
                        # new speech turn
                        if len(current_sentence['tokens']) > 0:
                            assert len(current_sentence['tokens']) == len(current_sentence['ner_tags'])
                            data.append(current_sentence)
                            current_sentence = {"tokens": [], "ner_tags": []}
                    else:
                        current_sentence["tokens"].append(token)
                        current_sentence["ner_tags"].append(tag)
            if current_sentence["tokens"]:
                assert len(current_sentence['tokens']) == len(current_sentence['ner_tags'])
                data.append(current_sentence)

        # Remove the original file after reading
        os.remove(file_path)
        logging.info(f"Deleted CONLL file: {file_path}")

        # Shuffle and split data
        df = adjust_conll_data(pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True))
        save_conll_data(df, os.path.join(folder, file_map.get(f)))


def preprocess_relation_classification(folder):
    """
    Preprocess relation classification data from TSV format to CSV.
    Assumes the TSV files are named latest_train.tsv, latest_dev.tsv, and latest_test.tsv.
    Drops duplicates and nulls, renames 'merged_sent' to 'text'.
    """
    split_dict = {
        'dev': 'latest_dev.tsv',
        'train': 'latest_train.tsv',
        'test': 'latest_test.tsv'
    }

    for split, f in split_dict.items():
        # Load TSV into DataFrame
        df = pd.read_csv(os.path.join(folder, f), sep='\t')
        df.rename({'merged_sent': 'text'}, inplace=True, axis=1)
        df = df.dropna().drop_duplicates().reset_index(drop=True)

        # Save cleaned DataFrame as CSV
        csv_path = os.path.join(folder, f"{split}.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"{split} dataset saved as CSV at {csv_path}")

        # Remove original TSV file
        os.remove(os.path.join(folder, f))
        logging.info(f"Deleted TSV file: {os.path.join(folder, f)}")


def preprocess_ner(folder):
    """
    Preprocess standard NER data from CONLL-style format.
    Each line contains a token and a tag, and sentences are separated by blank lines.
    Outputs train/dev/test splits in JSON format.
    """
    for f in ["dev", "test", "train"]:
        data = []
        file_path = os.path.join(folder, f"{f}.txt")
        current_sentence = {"tokens": [], "ner_tags": []}

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    token, tag = line.split('\t')
                    token = unidecode(token)
                    current_sentence["tokens"].append(token)
                    current_sentence["ner_tags"].append(tag)
                else:
                    data.append(current_sentence)
                    current_sentence = {"tokens": [], "ner_tags": []}
            if current_sentence["tokens"]:
                data.append(current_sentence)

        # Remove the original file after reading
        os.remove(file_path)
        logging.info(f"Deleted CONLL file: {file_path}")

        # Shuffle and split data
        df = adjust_conll_data(pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True))

        # Avoid duplicates
        df['tokens_tuple'] = df['tokens'].apply(tuple)
        df['ner_tags_tuple'] = df['ner_tags'].apply(tuple)
        df = df.drop_duplicates(subset=['tokens_tuple', 'ner_tags_tuple'])
        df = df.drop(columns=['tokens_tuple', 'ner_tags_tuple'])
        df = df.dropna().reset_index(drop=True)
        # Save splits
        save_conll_data(df, os.path.join(folder, f"{f}.json"))


def preprocess_sentiment_analysis(folder):
    """
    Preprocess sentiment analysis dataset.
    Assumes there is a single CSV file containing 'speech' and 'vote' columns.
    Renames them to 'text' and 'label', splits into train/dev/test, and saves.
    """
    file_path = os.path.join(folder, os.listdir(folder)[0])
    df = pd.read_csv(file_path)[['speech', 'vote']]
    df.rename({"speech": "text", "vote": "label"}, inplace=True, axis=1)

    # Split the dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    test, dev = train_test_split(test, test_size=0.5, random_state=42)

    # Save to CSV
    train.to_csv(os.path.join(folder, "train.csv"), index=False)
    test.to_csv(os.path.join(folder, "test.csv"), index=False)
    dev.to_csv(os.path.join(folder, "dev.csv"), index=False)

    # Remove original file
    os.remove(file_path)


if __name__ == "__main__":
    root = rootutils.find_root("")
    preprocess_argument_detection(os.path.join(root, "data/argument_detection"))
    preprocess_relation_classification(os.path.join(root, "data/relation_classification"))
    preprocess_ner(os.path.join(root, "data/ner"))
    preprocess_sentiment_analysis(os.path.join(root, "data/sentiment_analysis"))
