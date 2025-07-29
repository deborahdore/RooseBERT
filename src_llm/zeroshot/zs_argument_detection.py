import argparse
import logging
import os
import re
import warnings

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from utils import flatten, load_data

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

prompt = (
    "You are an argument analysis assistant. Your task is to identify and label argumentative spans in the input sentence according to the following types:\n"
    "- <claim>: expresses a stance, opinion, or proposed policy\n"
    "- <premise>: provides justification or support for a claim\n\n"
    "Instructions:\n"
    "1. Rewrite the entire sentence *without* changing or omitting any words.\n"
    "2. Surround each identified span with the appropriate tag: <claim>...</claim> or <premise>...</premise>.\n"
    "3. Tag only complete spans of meaning — do not tag partial words or sentence fragments.\n"
    "4. Do not tag anything that does not clearly fit one of the defined types.\n"
    "5. Output *only* the fully tagged sentence. Do not add any explanations or commentary.\n\n"
    "Sentence: %s"
)


def transform_into_tags(original_sentence: str, predicted_tokens: str, length: int):
    """
    Produces BIO tags aligned with the original sentence, even if only partial matches
    of tagged spans are found in the original tokens.
    """
    original_tokens = original_sentence.strip().split()
    bio_labels = ['O'] * len(original_tokens)

    # Match tagged entities in predicted output
    pattern = re.compile(r"<(claim|premise)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
    spans = pattern.findall(predicted_tokens)

    for tag, span_text in spans:
        tag = tag.lower()
        span_tokens = span_text.strip().split()
        used_indices = set()

        for span_tok in span_tokens:
            # Try to find first unmatched occurrence of span_tok in original_tokens
            for i, orig_tok in enumerate(original_tokens):
                if i in used_indices:
                    continue
                if orig_tok == span_tok:
                    used_indices.add(i)
                    break  # Stop after first match

        # Apply BIO labels in order of appearance
        sorted_indices = sorted(used_indices)
        for idx, token_idx in enumerate(sorted_indices):
            bio_labels[token_idx] = f"{'B' if idx == 0 else 'I'}-{tag}"

    return bio_labels[:length]


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
            trust_remote_code=True,
            device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        do_sample=False,
        max_new_tokens=512,
        repetition_penalty=1.1,
        torch_dtype=torch.bfloat16,
    )

    prompts = dataset["prompt"].tolist()
    sentences = dataset["sentence"].tolist()
    gold_labels = dataset["ner_tag"].apply(lambda x: x.split() if isinstance(x, str) else []).tolist()
    assert len(sentences) == len(gold_labels)

    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_sentences = sentences[i:i + batch_size]
        batch_labels = gold_labels[i:i + batch_size]

        with torch.inference_mode():
            outputs = generator(batch_prompts)

        for j, output_dict in enumerate(outputs):
            results.append({
                "sentence": batch_sentences[j],
                "ner_tags": batch_labels[j],
                "response": output_dict[0]["generated_text"]
            })

    results = pd.DataFrame(results).reset_index(drop=True)
    results.to_csv(results_file, index=False)

    response_ner_tags = []
    for sentence, response, ner_tag in zip(results['sentence'].tolist(), results['response'].tolist(),
                                           results['ner_tags'].tolist()):
        response_ner_tags.append(
            transform_into_tags(original_sentence=sentence, predicted_tokens=response, length=len(ner_tag)))

    flat_preds = flatten(response_ner_tags)
    flat_labels = flatten(results['ner_tags'])

    assert len(flat_preds) == len(flat_labels), "Prediction and label length mismatch."

    return {
        'accuracy': accuracy_score(flat_labels, flat_preds),
        'precision': precision_score(flat_labels, flat_preds, average="macro"),
        'recall': recall_score(flat_labels, flat_preds, average="macro"),
        'f1': f1_score(flat_labels, flat_preds, average="macro"),
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

    task = "argument_detection"
    dataset = load_data(f"data/{task}/test.json")
    dataset["prompt"] = dataset["sentence"].apply(lambda x: prompt % x)

    output_dir = os.path.join("logs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "argument_detection_predictions.csv")

    metrics = main(model_id="/lustre/fsmisc/dataset/HuggingFace_Models/" + model_id, dataset=dataset,
                   results_file=results_file)

    logging.info("###################### RESULTS ######################")
    logging.info(f"Model: {model_name}")
    for k, v in metrics.items():
        logging.info(f"{k.capitalize()}: {v}")
    logging.info("######################################################")
