"""
Preprocessing Script: U.S. Presidential Debates from UCSB Archive

This script downloads and processes transcripts of public presidential debates
from the UCSB Presidential Documents Archive.

Source:
https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0

Output:
Saves a cleaned CSV containing debate texts, URLs, titles, and dates.

Usage:
    Run the script directly — no arguments needed.
"""

import os
import re

import pandas as pd
import requests
import rootutils
from bs4 import BeautifulSoup
from tqdm import tqdm

from script.utils import clean_text, convert_to_dmy_format

# Constants
DEBATES_URL = "https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0"

# Setup project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Output path
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(""), "data", "training")
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "us_presidential_debates.csv")

# Regex to detect speaker prefixes like "BIDEN:" or "TRUMP:"
name_pattern = re.compile(r"^[A-Z]+:")


def fetch_url_content(url: str):
    """
    Fetch and parse HTML content from a URL.
    Returns a BeautifulSoup object or None if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None


def extract_debate_text(content_div):
    """
    Extract and structure paragraphs from a debate's HTML content.
    Groups text by speaker, skipping moderator/participant labels.
    """
    processed_texts = []
    current_text = ""

    for paragraph in content_div.find_all("p"):
        text = paragraph.get_text(strip=True, separator=" ").strip()

        if name_pattern.match(text):
            prefix = name_pattern.match(text).group(0)

            # Skip non-speech blocks
            if prefix in {'PARTICIPANTS:', 'MODERATORS:'}:
                continue

            # Save current block and start new one
            if current_text:
                processed_texts.append(current_text)
            current_text = re.sub(r"^[A-Z]+:\s*", "", text).strip()
        else:
            current_text += " " + text

    if current_text:
        processed_texts.append(current_text)

    return processed_texts


def main(main_page_url: str, path_to_csv: str):
    """
    Main function to scrape debate data from UCSB and save as CSV.
    """
    os.makedirs(os.path.dirname(path_to_csv), exist_ok=True)
    debate_data = []

    # Load index page containing debate links
    soup = fetch_url_content(main_page_url)
    if not soup:
        return

    # Extract all table rows (one per debate)
    debate_rows = soup.find_all("tr")
    print(f"📄 Found {len(debate_rows)} potential debate entries")

    for row in tqdm(debate_rows, desc="Scraping debates"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Extract date and title
        date = cells[0].get_text(strip=True)
        title_cell = cells[1]
        title = title_cell.get_text(strip=True)

        # Extract debate URL
        link_tag = title_cell.find("a")
        if not link_tag or not link_tag.get("href"):
            continue

        url = link_tag["href"]
        debate_soup = fetch_url_content(url)
        if not debate_soup:
            continue

        # Extract main content block
        content_div = debate_soup.find("div", class_="field-docs-content")
        if not content_div:
            continue

        # Parse and group text blocks by speaker
        processed_texts = extract_debate_text(content_div)

        # Save each speech block as a separate record
        for processed_text in processed_texts:
            debate_data.append({
                "text": processed_text,
                "url": url,
                "title": title,
                "date": convert_to_dmy_format(date, current_format="%B %d, %Y"),
            })

    # Convert to DataFrame and export to CSV
    df = pd.DataFrame(debate_data)
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df.to_csv(path_to_csv, index=False)
    print("Dataset Length: {}".format(len(df)))
    print(f"✅ Debate data saved to: {path_to_csv}")


if __name__ == '__main__':
    main(DEBATES_URL, PATH_TO_FINAL_FILE)
    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
