import json
import os
import warnings

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline

# Setup
warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def build_prompt(sentence):
    return f"""[INST]
    You are a helpful assistant. Your task is to classify the **relation** between two arguments in the sentence below. The arguments are separated by a dot (".").
    
    Instructions:
    - Classify the relation between the two arguments as one of the following **exactly**:
      - support
      - attack
      - equivalent
    - Return **only** one of these words: support, attack, or equivalent.
    - Do **not** include any explanation, punctuation, or additional text.
    - Your output must be **only** the classification word.
    
    Sentence:
    "{sentence}"
    [/INST]"""


if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    dataset = pd.read_csv("data/relation_classification/test.csv")
    dataset["prompt"] = dataset["text"].apply(build_prompt)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        do_sample=False,
        max_new_tokens=100,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    preds = []
    labels_cleaned = []
    for row_idx, row in enumerate(tqdm(dataset.itertuples(index=False), total=len(dataset))):

        try:
            with torch.inference_mode():
                output = generator(row.prompt, max_new_tokens=100)[0]["generated_text"].replace("\n", "").lower()

            assert "support" in output or "attack" in output or "equivalent" in output

            if "support" in output:
                preds.append(0)
            elif "attack" in output:
                preds.append(1)
            else:
                preds.append(2)

            labels_cleaned.append(row.link_type)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[{row_idx}] Format error: {e}\n Output: {output}")
        finally:
            continue

    assert len(preds) == len(labels_cleaned)

    print(
        {'accuracy': accuracy_score(y_true=labels_cleaned, y_pred=preds),
         'precision': precision_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
         'recall': recall_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
         'f1': f1_score(y_true=labels_cleaned, y_pred=preds, average="macro")
         }
    )
