import re
from datetime import datetime

import pandas as pd


def clean_text(content):
    """Cleans text content by removing unwanted characters and formatting issues."""
    content = str(content)
    # Remove HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    # Normalize whitespace characters to single spaces
    content = re.sub(r'[\t\n\r\f\v]+', ' ', content)
    # Remove numbering patterns like "1:" or "1." at the beginning of lines
    content = re.sub(r'^\d+[:.]\s*', '', content, flags=re.MULTILINE)
    # Collapse multiple spaces into one
    content = re.sub(r' {2,}', ' ', content)
    # Replace common title abbreviations that include a period (optional)
    abbreviations = {
        "Mr.": "Mr", "Mrs.": "Mrs", "Ms.": "Ms",
        "Dr.": "Dr", "Prof.": "Prof", "Sir.": "Sir",
        "Hon.": "Hon", "hon.": "hon"
    }
    for abbr, clean in abbreviations.items():
        content = content.replace(abbr, clean)
    return content.strip()


def load_file_body(file_path):
    """Load and concatenate all body text from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        phrases = df['body'].dropna().astype(str).tolist()
        return " ".join(phrases)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def count_total_words(df):
    """ Counts the total number of words in the 'text' column of a DataFrame. """
    return df.apply(lambda x: len(str(x).split())).sum()


def convert_to_dmy_format(date_str, current_format):
    """
    Converts a date string from the given format to 'd-m-Y' format.

    Parameters:
    - date_str (str): The date string to convert.
    - current_format (str): The format of the input date string.

    Returns:
    - str: The date in 'd-m-Y' format.
    """
    try:
        date_obj = datetime.strptime(date_str, current_format)
        return date_obj.strftime("%d-%m-%Y")
    except ValueError as e:
        return f"Error: {e}"
