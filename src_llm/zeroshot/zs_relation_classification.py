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


def build_prompt(sentence, model_name):
    prompt = f"""Classify the **relation** between two arguments in the sentence below. The arguments are separated by a period (".").

    Allowed relation types:
    - support
    - attack
    - neither

    Instructions:
    - Return **only** one of the allowed relation types: support, attack, or neither if no relation is found.
    - Do **not** include any explanation, punctuation, or extra text.
    - Output must be **only** the relation word, exactly as written.

    Sentence:
    "%s" """
    prompt_ = prompt % sentence
    if model_name == "Mistral-7B-Instruct-v0.3" or model_name == "Llama-3.1-8B-Instruct":
        return f"[INST] {prompt_} [/INST]"
    elif model_name == "gemma-3-4b-it":
        return (f"<start_of_turn>user "
                f"{prompt_} <end_of_turn>"
                f"<start_of_turn>model")


def main(model_id: str, dataset: pd.DataFrame, batch_size: int = 8):
    # Load model and tokenizer
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
        max_new_tokens=25,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16
    )

    prompts = dataset["prompt"].tolist()
    sentences = dataset["text"].tolist()
    gold_labels = dataset["link_type"].tolist()
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

                assert "support" in output or "attack" in output or "neither" in output or "none" in output or "no relation" in output
                if "support" in output:
                    preds.append(0)
                elif "attack" in output:
                    preds.append(1)
                else:
                    preds.append(2)
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

    dataset = pd.read_csv("data/relation_classification/test.csv")
    dataset["prompt"] = dataset["text"].apply(lambda x: build_prompt(x, model_name=model_name))

    results = []
    models_results = main("/lustre/fsmisc/dataset/HuggingFace_Models/" + model_id, dataset)

    results.append({
        'model': model_name,
        'accuracy': models_results['accuracy'],
        'precision': models_results['precision'],
        'recall': models_results['recall'],
        'f1': models_results['f1'],
        'errors': models_results['errors']
    })

    print("################################### RESULTS ###################################")
    print("Model:", model_name)
    print(results)
    print("###############################################################################")

    with pd.ExcelWriter(os.path.join(rootutils.find_root(""), f"results_{model_name}.xlsx")) as writer:
        df = pd.DataFrame(results)
        df.to_excel(writer, sheet_name="relation_classification", index=False)
