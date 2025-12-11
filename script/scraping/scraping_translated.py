import os.path

import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

INPUT_FOLDER = "~/Downloads"
OUTPUT_FILE = "./data/training/translated.csv"


def main(file):
    dataset = []
    try:
        df_info = pd.read_csv(file, delimiter="\t", names=['id', 'text'])
        df_meta = pd.read_csv(file[:-4] + "-meta.tsv", delimiter="\t")
        for idx, row in df_info.iterrows():
            info_id = row['id']
            speech = row['text']
            meta = df_meta[df_meta['ID'] == info_id]
            speaker = meta.Speaker_ID.item()
            date = meta.Date.item()

            dataset.append({
                "ID": info_id.split(".")[0],
                "date": date,
                "speaker": speaker,
                "text": speech,
            })
        if len(dataset) > 0:
            return pd.DataFrame(dataset).dropna().drop_duplicates().reset_index(drop=True)
        return pd.DataFrame()
    except Exception as e:
        print("Cannot process file", file)
        return pd.DataFrame()


if __name__ == "__main__":
    input_folder = os.path.expanduser(INPUT_FOLDER)
    parlamint_dirs = [d for d in os.listdir(input_folder) if
                      d.startswith("ParlaMint") and os.path.isdir(os.path.join(input_folder, d)) and d.endswith(".ana")]
    dataframes = []
    for d in tqdm(parlamint_dirs, total=len(parlamint_dirs), desc="Processing ParlMint"):
        base_dir = os.path.join(input_folder, d)
        txt_root = os.path.join(base_dir, d.replace(".ana", ".txt"))

        if not os.path.isdir(txt_root):
            continue

        # Year subdirectories
        year_dirs = [
            y for y in os.listdir(txt_root)
            if os.path.isdir(os.path.join(txt_root, y))
        ]

        for year in year_dirs:
            year_path = os.path.join(txt_root, year)

            # .txt files in each year directory
            txt_files = [
                f for f in os.listdir(year_path)
                if f.endswith(".txt")
            ]

            for f in txt_files:
                file_path = os.path.join(year_path, f)
                data = main(file_path)
                if len(data) > 0:
                    dataframes.append(data)

    df = pd.concat(dataframes, axis=0).dropna().drop_duplicates().reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    os.system(f"du -sh {OUTPUT_FILE}")
