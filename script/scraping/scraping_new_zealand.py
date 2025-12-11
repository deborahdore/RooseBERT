"""
Preprocessing Script for New Zealand
Download: https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/L4OAKN/LLMYON&version=1.0
"""
import os

import rootutils
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

from script.utils import clean_text

# Set up project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Define output path
PATH_TO_FINAL_FOLDER = os.path.join(rootutils.find_root(), "data", "training")
os.makedirs(PATH_TO_FINAL_FOLDER, exist_ok=True)
PATH_TO_FINAL_FILE = os.path.join(PATH_TO_FINAL_FOLDER, "newZealand.csv")


def main(input_file: str, output_file: str):
    print(f"✅ Opening dataset")
    readRDS = ro.r['readRDS']
    obj = readRDS(input_file)

    # Force conversion to a base data.frame in R
    obj_base = ro.r('as.data.frame')(obj)
    df = pandas2ri.rpy2py(obj_base)
    df['ID'] = "NewZealandParliament_" + df['date']
    df = df[['ID', 'date', 'speaker', 'text']]
    df["text"] = df["text"].apply(clean_text)
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print("Dataset Length: ", len(df))
    print(f"✅ Processed dataset saved to: {output_file}")


if __name__ == "__main__":
    main(os.path.expanduser("~/Downloads/Corp_NZHoR_V2.rds"), PATH_TO_FINAL_FILE)
    os.system(f"du -sh {PATH_TO_FINAL_FILE}")
