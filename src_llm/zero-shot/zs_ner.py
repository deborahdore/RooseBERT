import json
import os
import re
import warnings

import pandas as pd
import rootutils
import torch
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline

# Setup
warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def load_data(filepath):
    with open(filepath, 'r') as f:
        raw_data = json.load(f)
    records = []
    for entry in raw_data:
        sentence = " ".join(entry['tokens']).strip()
        ner_tag = " ".join(entry['ner_tags']).strip().lower()
        records.append({
            "sentence": sentence,
            "ner_tag": ner_tag
        })
    return pd.DataFrame(records)


def build_prompt(sentence):
    prompt = """[INST]
    You are a helpful assistant. Your task is to perform Named Entity Recognition (NER) on the sentence below.

    Return each identified named entity as a **separate, valid JSON object**.

    Allowed entity types:
    - politician
    - person
    - organization
    - politicalparty
    - event
    - election
    - country
    - location
    - miscellaneous

    Instructions:
    - For each entity, return a JSON object with exactly two fields:
      - "sentence": the exact text span from the input sentence representing the entity.
      - "type": one of the allowed entity types above.
    - If no entity is found, return a single JSON object with both fields as empty strings: `{{"sentence": "", "type": ""}}`
    - Do **not** wrap the JSON objects in a list (i.e., no square brackets).
    - Separate multiple JSON objects with **commas and a space only**, like this:
      `{{"sentence": "...", "type": "..."}}, {{"sentence": "...", "type": "..."}}`
    - The output must be **strictly valid JSON**:
      - Use double quotes
      - No trailing commas
      - All braces must be correctly closed
    - Do **not** include any explanation, commentary, or extra text. Output only the JSON objects.

    Sentence:
    "%s"
    [/INST]"""
    return prompt % sentence


if __name__ == "__main__":
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    dataset = load_data("data/ner/test.json")
    dataset["prompt"] = dataset["sentence"].apply(build_prompt)

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
        max_new_tokens=500,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    preds = []
    labels_cleaned = []
    labels = dataset["ner_tag"].apply(lambda x: x.split() if isinstance(x, str) else []).tolist()
    for row_idx, row in enumerate(tqdm(dataset.itertuples(index=False), total=len(dataset))):

        original_sentence = row.sentence.lower().strip().split()
        ner_mask = ['o'] * len(original_sentence)

        try:
            with torch.inference_mode():
                output = generator(row.prompt)[0]["generated_text"].replace("\n", "")
                output = re.sub(r'^.*?{', '{', output)
                output = re.sub(r'}[^}]*$', '}', output)
                output = output.replace("\\", "")
                output = f"[{output}]"

            parsed_data = json.loads(output)

            for item in parsed_data:
                pred_sentence = item.get('sentence').lower().strip().split()
                pred_type = item.get('type').lower().strip()

                if len(pred_type.split()) == 0 or len(pred_sentence) == 0: continue
                if len(pred_sentence) > len(original_sentence): continue
                if item.get('sentence').lower().strip() not in row.sentence.lower().strip(): continue

                first_token = pred_sentence[0]
                len_pred_sentence = len(pred_sentence)

                idx = original_sentence.index(first_token)
                ner_mask[idx] = f"b-{pred_type}"
                for i in range(1, len_pred_sentence):
                    ner_mask[idx + i] = f"i-{pred_type}"

            preds.append(ner_mask)
            labels_cleaned.append(labels[row_idx])  # Only add label if parsing succeeded

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[{row_idx}] Format error: {e}\n Output: {output}")
        finally:
            continue

    assert len(preds) == len(labels_cleaned)
    print(
        {'accuracy': accuracy_score(y_true=labels_cleaned, y_pred=preds),
         'precision': precision_score(y_true=labels_cleaned, y_pred=preds, average="macro").item(),
         'recall': recall_score(y_true=labels_cleaned, y_pred=preds, average="macro").item(),
         'f1': f1_score(y_true=labels_cleaned, y_pred=preds, average="macro").item()
         }
    )
