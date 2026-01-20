"""
Preprocessing Script: Canadian Parliamentary Debates (OpenParliament)

This script downloads and processes transcripts of debates from:
    https://openparliament.ca/debates/

Output:
    Saves a cleaned CSV containing debate texts, URLs, speakers, and dates.

Usage:
    Run the script directly — no arguments needed.
"""

import os
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
import rootutils
from bs4 import BeautifulSoup
from tqdm import tqdm

from script.utils import clean_text, convert_to_dmy_format

BASE_URL = "https://openparliament.ca"
DEBATES_URL = f"{BASE_URL}/debates/"
FROM = 1994
TO = 2025

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

OUTPUT_FOLDER = os.path.join(rootutils.find_root(""), "data", "training")
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "canada.csv")


def fetch_soup(url: str) -> BeautifulSoup | None:
    """Retrieve URL and return BeautifulSoup object, or None on failure."""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"⚠️  Failed to fetch {url}: {e}")
        return None


def extract_text_block(div: BeautifulSoup) -> str:
    """Extract text from <p> nodes inside a speech block."""
    paragraphs = div.find_all("p")
    return " ".join(
        p.get_text(" ", strip=True)
        for p in paragraphs
    ).strip()


def find_statement_blocks(soup: BeautifulSoup):
    """Find all statement blocks within the paginated debate page."""
    container = soup.find("div", id="paginated")
    if not container:
        return []

    return container.find_all(
        "div",
        class_=lambda cls: (
                cls
                and "row" in cls
                and "statement_browser" in cls
                and "statement" in cls
        )
    )


def main(index_url: str, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_rows = []

    # Single tqdm bar for the year loop
    for year in range(FROM, TO + 1):

        year_url = urljoin(index_url, f"{year}/")
        year_soup = fetch_soup(year_url)
        if not year_soup:
            continue

        # Debate links for this year
        debate_links = year_soup.find_all(
            "a",
            href=re.compile(rf"^/debates/{year}/")
        )

        for a in tqdm(debate_links, desc=f"Scraping {year}", unit="debate"):
            try:

                # Parse the date
                raw_date = a.get_text(strip=True)
                clean_date = re.sub(r"(st|nd|rd|th)", "", raw_date)

                date = convert_to_dmy_format(clean_date, "%B %d")

                # Load debate page
                debate_url = urljoin(BASE_URL, a["href"]) + "?singlepage=1"
                debate_soup = fetch_soup(debate_url)
                if not debate_soup:
                    continue

                # Extract statement blocks
                blocks = find_statement_blocks(debate_soup)
                if not blocks:
                    continue

                # Extract text from each block
                for block in blocks:
                    speaker_tag = block.find("span", class_="pol_name")
                    speaker = speaker_tag.get_text(strip=True) if speaker_tag else "Procedural"

                    text_div = block.find("div", class_="text")
                    if not text_div:
                        continue

                    text = extract_text_block(text_div)

                    all_rows.append({
                        "ID": f"CanadaParliament_{date}",
                        "date": date,
                        "speaker": speaker,
                        "text": text,
                    })
            except Exception as e:
                time.sleep(5)
                continue

    print("\n Cleaning dataset...")
    df = pd.DataFrame(all_rows)

    df["text"] = df["text"].astype(str).apply(clean_text)

    df = df.dropna().drop_duplicates().reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved {len(df)} debate rows to {output_file}")


if __name__ == "__main__":
    main(DEBATES_URL, OUTPUT_FILE)
    os.system(f"du -sh {OUTPUT_FILE}")
