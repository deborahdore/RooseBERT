import argparse
import os
import re
import warnings

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Setup
warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

LABEL2ID_MAP = {'support': 0, 'attack': 1, 'no_relation': 2}
ID2LABEL_MAP = {0: 'support', 1: 'attack', 2: 'no_relation'}
ALLOWED_LABELS = {"support", "attack", "neither", "none", "no relation"}

prompt = (
    "You are a relation classification assistant. Your task is to determine the type of argumentative relation between two argument spans in the sentence below. "
    "The two arguments are separated by a period.\n\n"
    "Output instructions:\n"
    "- Output only one of the allowed relation types: support, attack, or neither\n"
    "- Do not include punctuation, explanation, or formatting\n\n"
    "Examples:\n"
    "Sentence: We're not going to support the $300 billion tax cut that they have for corporate America and the very wealthy. It didn't meet my test.\n"
    "Output: neither\n\n"
    "Sentence: They do have stakes in it. What we need now is a president who understands how to bring these other countries together to recognize their stakes in this.\n"
    "Output: support\n\n"
    "Sentence: I respect the belief about life and when it begins. I can't take what is an article of faith for me and legislate it for someone who doesn't share that article of faith, whether they be agnostic, atheist, Jew, Protestant, whatever.\n"
    "Output: attack\n\n"
    "Sentence: %s"
)


def generate_predictions(generator, prompts, gold_labels, batch_size: int = 8):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_labels = gold_labels[i:i + batch_size]

        with torch.inference_mode():
            outputs = generator(batch_prompts)

        for j, output_dict in enumerate(outputs):
            output = output_dict[0]["generated_text"]
            results.append({
                'prompt': batch_prompts[j],
                'y_true': ID2LABEL_MAP[batch_labels[j]],
                'y_pred': output
            })

    return pd.DataFrame(results).reset_index(drop=True)


def normalize_prediction(output: str) -> str:
    output = output.replace("\n", "").lower()
    if "[/inst]" in output:
        output = re.sub(r'\s*\[/inst\].*', '', output, flags=re.DOTALL)
    return output


def main(model_id: str, dataset: pd.DataFrame, results_file: str, batch_size: int = 8):
    if "gemma" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
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
        max_new_tokens=50,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    results_df = generate_predictions(generator, dataset["prompt"].tolist(), dataset["link_type"].tolist(), batch_size)
    results_df.to_csv(results_file, index=False)

    preds, labels_cleaned = [], []
    errors = 0

    for _, row in results_df.iterrows():
        y_true, raw_output = row['y_true'], row['y_pred']
        try:
            output = normalize_prediction(raw_output)
            if not any(label in output for label in ALLOWED_LABELS):
                raise ValueError("No valid label found in output.")

            if "support" in output:
                preds.append(LABEL2ID_MAP["support"])
            elif "attack" in output:
                preds.append(LABEL2ID_MAP['attack'])
            else:
                preds.append(LABEL2ID_MAP['no_relation'])
            labels_cleaned.append(LABEL2ID_MAP[y_true])

        except Exception:
            errors += 1

    if len(preds) != len(labels_cleaned):
        raise ValueError("Mismatch between predictions and ground truth.")

    return {
        'accuracy': accuracy_score(labels_cleaned, preds),
        'precision': precision_score(labels_cleaned, preds, zero_division=0),
        'recall': recall_score(labels_cleaned, preds, zero_division=0),
        'f1': f1_score(labels_cleaned, preds, zero_division=0),
        'errors': errors
    }


if __name__ == "__main__":
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "google/gemma-3-4b-it"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier or path")
    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[-1]

    dataset = pd.read_csv("data/relation_classification/test.csv")
    dataset["prompt"] = dataset["text"].apply(lambda x: prompt % x)
    results_file = "logs/" + model_name + "/relation_classification_predictions.csv"
    os.makedirs("logs/" + model_name, exist_ok=True)

    metrics = main("/lustre/fsmisc/dataset/HuggingFace_Models/" + model_id, dataset, results_file)

    print("\n############## RESULTS ##############")
    print(f"Model: {model_id}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Errors:    {metrics['errors']}")
    print("#####################################\n")
