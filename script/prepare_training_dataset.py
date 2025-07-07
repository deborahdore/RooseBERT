"""
Script used to prepare data for the masked language modelling task.
"""

import logging
import os

import nltk
import pandas as pd
import rootutils
from nltk import sent_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')

# Setup project root and data directory
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
BASE_PATH = rootutils.find_root(search_from=__file__, indicator=".project-root")
DATA_DIR = os.path.join(BASE_PATH, 'data/training')


def split_sentences(series: pd.Series) -> pd.DataFrame:
    """
    Split each text entry in the Series into individual sentences.

    Args:
        series (pd.Series): A pandas Series of text entries.

    Returns:
        pd.DataFrame: A DataFrame with each sentence in its own row under column 'text'.
    """
    logger.debug("Splitting text into sentences...")
    return (
        series.fillna("")
        .astype(str)
        .apply(sent_tokenize)
        .explode()
        .reset_index(drop=True)
        .to_frame(name="text")
    )


def concatenate_in_chunks(df: pd.DataFrame, max_sequence: int) -> pd.DataFrame:
    """
    Concatenate sentences into chunks without exceeding max_sequence words.

    Args:
        df (pd.DataFrame): DataFrame with a 'text' column.
        max_sequence (int): Maximum words per chunk.

    Returns:
        pd.DataFrame: Chunks of text in a DataFrame.
    """
    logger.debug("Concatenating sentences into chunks of max %d words", max_sequence)

    chunks = []
    current_chunk = []
    chunks_size = 0

    sentences = df['text'].dropna().tolist()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_size = len(sentence.split())

        if chunks_size + sentence_size + 1 <= max_sequence:
            current_chunk.append(sentence)
            chunks_size += sentence_size
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks_size = sentence_size
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.debug("Created %d chunks", len(chunks))
    return pd.DataFrame({'text': chunks})


def load_and_process_data():
    """
    Load all CSVs, split sentences, chunk text, and split into train/dev.

    Args:
        max_sequence (int): Max number of words per chunk.

    Returns:
        Combined train and dev sets.
    """
    all_train, all_dev, all_test = [], [], []

    files = [f for f in os.listdir(DATA_DIR) if
             (f.endswith(".csv") and (not f.startswith("train")) and (not f.startswith("dev")))]
    logger.info("Found %d CSV files in %s", len(files), DATA_DIR)

    for file in tqdm(files, desc="Processing files"):
        file_path = os.path.join(DATA_DIR, file)
        logger.info("Processing file: %s", file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error("Failed to read %s: %s", file, e)
            raise

        if 'text' not in df.columns:
            logger.warning("Skipping %s: No 'text' column found", file)
            raise

        # Process text
        df = split_sentences(df.pop('text'))

        # Split into train and dev
        train, dev = train_test_split(df, test_size=0.1, random_state=42, shuffle=False)
        #  held out dataset for perplexity
        dev, test = train_test_split(dev, test_size=0.01, random_state=42, shuffle=False)

        all_train.append(train)
        all_dev.append(dev)
        all_test.append(test)

    if not all_train or not all_dev:
        logger.error("No data processed. Exiting.")
        raise ValueError("No data processed.")

    train = pd.concat(all_train).reset_index(drop=True)
    dev = pd.concat(all_dev).reset_index(drop=True)
    test = pd.concat(all_test).reset_index(drop=True)

    train_128 = concatenate_in_chunks(train, 128)
    train_512 = concatenate_in_chunks(train, 512)

    dev_128 = concatenate_in_chunks(dev, 128)
    dev_512 = concatenate_in_chunks(dev, 512)

    test_512 = concatenate_in_chunks(test, 512)

    return shuffle(train_128), shuffle(train_512), shuffle(dev_128), shuffle(dev_512), shuffle(test_512)


def shuffle(df):
    return df.sample(frac=1).dropna().drop_duplicates().reset_index(drop=True)


def main():
    """
    Main workflow: Load, preprocess, split, save data.

    """
    train_128, train_512, dev_128, dev_512, test_512 = load_and_process_data()

    # Log dataset sizes
    logger.info("Train set size: %d", len(train_128))
    logger.info("Dev set size: %d", len(dev_128))
    logger.info("Total processed: %d", len(train_128) + len(dev_128))

    # Save output
    for max_sequence in [128, 512]:
        os.makedirs(os.path.join(DATA_DIR, f"max_{max_sequence}"), exist_ok=True)

    train_128.to_csv(os.path.join(DATA_DIR, 'max_128/train.csv'), index=False)
    dev_128.to_csv(os.path.join(DATA_DIR, 'max_128/dev.csv'), index=False)

    train_512.to_csv(os.path.join(DATA_DIR, 'max_512/train.csv'), index=False)
    dev_512.to_csv(os.path.join(DATA_DIR, 'max_512/dev.csv'), index=False)

    test_512.to_csv(os.path.join(DATA_DIR, 'perplexity_test.csv'), index=False)

    os.system(f"du -sh {os.path.join(DATA_DIR, 'max_128/train.csv')}")
    os.system(f"du -sh {os.path.join(DATA_DIR, 'max_128/dev.csv')}")
    os.system(f"du -sh {os.path.join(DATA_DIR, 'perplexity_test.csv')}")


if __name__ == '__main__':
    main()
