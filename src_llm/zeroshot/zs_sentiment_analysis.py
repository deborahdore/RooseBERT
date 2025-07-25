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

from utils import generate_prompt

# Setup
warnings.filterwarnings("ignore")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

# POSITIVE -> 1, NEGATIVE -> 0 in this dataset
label_map = {'positive': 1, 'negative': 0}


def build_prompt(model):
    role = "You are a sentiment classification assistant. Your task is to classify the sentiment of the sentence. The possibile sentiments are: positive or negative."
    instructions = """Output instructions:
        - Output only one of the allowed relation types: positive or negative.
        - Do not include punctuation, explanation, or formatting.
        """
    return generate_prompt(model, role, instructions)


def main(model_id: str, dataset: pd.DataFrame, batch_size: int = 8):
    # Load model and tokenizer
    if "google/gemma-3-4b-it" in model_id:
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
        max_new_tokens=100,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    prompts = dataset["prompt"].tolist()
    sentences = dataset["text"].tolist()
    gold_labels = dataset["label"].tolist()
    assert len(sentences) == len(gold_labels)

    preds = []
    labels_cleaned = []
    errors = 0
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_labels = gold_labels[i:i + batch_size]

        with torch.inference_mode():
            outputs = generator(batch_prompts)
        for j, output_dict in enumerate(outputs):
            output = output_dict[0]["generated_text"]
            try:
                if "[/INST]" in output:
                    output = re.sub(r'\s*\[/INST\].*', '', output, flags=re.DOTALL)
                output = output.replace("\n", "")
                output = output.lower()

                assert "positive" in output or "negative" in output

                if "positive" in output:
                    preds.append(label_map["positive"])
                else:
                    preds.append(label_map["negative"])
                labels_cleaned.append(batch_labels[j])

            except Exception as e:
                print(f"[Batch {i}] Generator error: {e}")
                print(f"Output: {output_dict[0]['generated_text']} \n \n")
                errors += 1
                continue

    assert len(preds) == len(labels_cleaned), "Mismatch between predictions and gold labels"

    return {
        'accuracy': accuracy_score(y_true=labels_cleaned, y_pred=preds),
        'precision': precision_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'recall': recall_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'f1': f1_score(y_true=labels_cleaned, y_pred=preds, average="macro"),
        'errors': errors
    }


if __name__ == "__main__":
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "google/gemma-3-4b-it"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier from Hugging Face")
    args = parser.parse_args()

    model_id = args.model_id
    model_name = model_id.split("/")[-1]

    task = "sentiment_analysis"
    dataset = pd.read_csv(f"data/{task}/test.csv")
    prompt = build_prompt(model_name)
    dataset["prompt"] = dataset["text"].apply(lambda x: prompt % x)

    results = []
    models_results = main("/lustre/fsmisc/dataset/HuggingFace_Models/" + model_id, dataset)

    results.append({
        'model': model_name,
        'task': task,
        'accuracy': models_results['accuracy'],
        'precision': models_results['precision'],
        'recall': models_results['recall'],
        'f1': models_results['f1'],
        'errors': models_results['errors']
    })

    print("################################### RESULTS ###################################")
    print("Model:", model_name)
    print("Task:", task)
    print(results)
    print("###############################################################################")

    file_path = os.path.join(rootutils.find_root(""), "results_zeroshot.xlsx")
    results_df = pd.DataFrame(results)

    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            try:
                existing_df = pd.read_excel(file_path, sheet_name=task)
                combined_df = pd.concat([existing_df, results_df], ignore_index=True).reset_index(drop=True)
            except ValueError:
                combined_df = results_df

            combined_df.to_excel(writer, sheet_name=task, index=False)

    else:
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name=task, index=False)
