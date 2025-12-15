import argparse
import os
import re
import warnings
from typing import Dict, List

import pandas as pd
import rootutils
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

# Setup
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

LABELS2ID = {
    "AusHansard": {'support': 1, 'oppose': 0},
    "ConVote": {'oppose': 0, 'support': 1},
    "ParlVote": {'positive': 1, 'negative': 0},
    "HanDeSet": {'positive': 1, 'negative': 0},
}

INSTRUCTION_PROMPT = {
    "HanDeSeT": (
        "You are a sentiment classification assistant. Classify the sentences using the following labels: positive, negative\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "ParlVote": (
        "You are a sentiment classification assistant. Classify the sentences using the following labels: positive, negative\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "ConVote": (
        "You are a stance classification assistant. Classify the sentences using the following labels: support, oppose\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "AusHansard": (
        "You are a stance classification assistant. Classify the sentences using the following labels: support, oppose\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),
}


def load_model(model_name: str, quantize_4bit: bool = True):
    """Load model + tokenizer with optional 4-bit quantization."""
    if quantize_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if quantize_4bit else None,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=False,
        temperature=0.0,
    )

    return pipe


def build_prompt(dataset: str, sentence: str) -> str:
    template = INSTRUCTION_PROMPT[dataset]
    return template.format(sentence=sentence)


def extract_label(text: str) -> str:
    """Extract the last word (e.g., 'support', 'oppose', 'positive', 'negative')."""
    match = re.search(r"(support|oppose|positive|negative)", text.lower())
    return match.group(1) if match else "none"


def compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(gold, pred),
        "precision": precision_score(gold, pred, average="macro", zero_division=0),
        "recall": recall_score(gold, pred, average="macro", zero_division=0),
        "f1": f1_score(gold, pred, average="macro", zero_division=0),
    }


def run(args):
    df = pd.read_csv(f"data/binary_classification/{args.dataset}/test.csv")
    pipe = load_model(args.model)

    predictions = []
    gold_labels = df[args.label_col].tolist()
    sentences = df[args.text_col].tolist()

    for s in tqdm(sentences, desc="Classifying"):
        prompt = build_prompt(args.dataset, s)
        output = pipe(prompt)[0]["generated_text"]
        label = extract_label(output)
        predictions.append(label)

    df["prediction"] = predictions
    output_path = f"logs/fs_binary_classification_{args.dataset}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to: {output_path}")

    metrics = compute_metrics(gold_labels, [LABELS2ID[args.dataset][x] for x in predictions])
    print("\nEvaluation:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str, choices=INSTRUCTION_PROMPT.keys(), default="ParlVote")

    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")

    args = parser.parse_args()
    run(args)
