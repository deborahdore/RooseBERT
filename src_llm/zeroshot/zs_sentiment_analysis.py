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
    return f"""[INST] Classify the **sentiment** of the following sentence.

    Allowed sentiment labels:
    - Positive
    - Negative

    Instructions:
    - Return **only** one of the two words: Positive or Negative.
    - Do **not** include any explanation, punctuation, or extra text.
    - Output must be exactly one word.

    Sentence:
    "{sentence}" [/INST]"""


def main(model_id: str, dataset: pd.DataFrame):
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
                output = generator(row.prompt)[0]["generated_text"].replace("\n", "").lower()

            assert "positive" in output or "negative" in output

            if "positive" in output:
                preds.append(1)
            else:
                preds.append(0)

            labels_cleaned.append(row.label)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[{row_idx}] Format error: {e}\n Output: {output}")
        finally:
            continue
    assert len(preds) == len(labels_cleaned)
    return {'accuracy': accuracy_score(y_true=labels_cleaned, y_pred=preds),
            'precision': precision_score(y_true=labels_cleaned, y_pred=preds),
            'recall': recall_score(y_true=labels_cleaned, y_pred=preds),
            'f1': f1_score(y_true=labels_cleaned, y_pred=preds)
            }


if __name__ == "__main__":
    model_ids = ["mistralai/Mistral-7B-Instruct-v0.3",
                 "meta-llama/Llama-3.1-8B-Instruct",
                 "google/gemma-7b-it"]
    output_file = os.path.join(rootutils.find_root(""), "results_llms.csv")

    dataset = pd.read_csv("data/sentiment_analysis/test.csv")
    dataset["prompt"] = dataset["text"].apply(build_prompt)

    results = []
    for model_id in model_ids:
        models_results = main(model_id, dataset)
        results.append({
            'model': model_id,
            'accuracy': models_results['accuracy'],
            'precision': models_results['precision'],
            'recall': models_results['recall'],
            'f1': models_results['f1']
        })
        print("Model:", model_id)
        print(results)

    with pd.ExcelWriter(output_file) as writer:
        df = pd.DataFrame(results)
        df.to_excel(writer, sheet_name="sentiment_analysis", index=False)
