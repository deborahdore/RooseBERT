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

INSTRUCTION_PROMPT = {
    "ElecDeb60to20-relations": (
        "You are a relation classification assistant. Classify the sentences separated by [SEP] using the labels: support, attack, no_relation\n\n"
        "Sentence: {sentence}\n"
        "Output:"
    ),
    "ArgUNSC": (
        "You are a relation classification assistant. Classify the sentences separated by [SEP] using the labels: support, attack, no_relation\n\n"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "MotionPolicyPreference": (
        "You are a motion policy preference classification assistant. Classify the motions using the following labels:"
        "Military: Positive, Military: Negative, Peace, Internationalism: Positive, "
        "European Union: Positive, European Union: Negative, Human Rights, "
        "Direct Democracy: Positive, Constitutionalism: Positive, Constitutionalism: Negative, "
        "Decentralisation: Positive, Centralisation: Positive, Governative and Administrative Efficiency, "
        "Political Corruption, Political Authority: Party, Free Market Economy, Incentives: Positive, "
        "Market Regulation, Economic Planning, Corporatism/Mixed Economy, Protectionism: Negative, "
        "Economic Goals, Keynesian Demand Management, Economic Growth: Positive, "
        "Technology and Infrastructure: Positive, Controlled Economy, Nationalisation, "
        "Economic Orthodoxy, Environmental Protection, Culture: Positive, Equality: Positive, "
        "Welfare State Expansion, Welfare State Limitation, Education Expansion, "
        "National Way of Life: Positive, National Way of Life: Negative, "
        "Law and Order: Positive, Civic Mindedness: Positive, Multiculturalism: Positive, "
        "Multiculturalism: Negative, Labour Groups: Positive, Agriculture and Farmers: Positive, "
        "Middle Class and Professional Groups, Underprivileged Minority Groups, "
        "Non-economic Demographic Groups\n\n"
        "Sentence: {sentence}\n"
        "Output:"
    ),

    "ParlVote+": (
        "You are a motion policy preference classification assistant. Classify the motions using the following labels:"
        "Military: Positive, Military: Negative, Peace, European Union: Positive, "
        "European Union: Negative, Human Rights, Direct Democracy: Positive, "
        "Constitutionalism: Positive, Constitutionalism: Negative, Decentralisation: Positive, "
        "Centralisation: Positive, Political Corruption, Political Authority: Party, "
        "Political Authority: Personal, Free Market Economy, Incentives: Positive, "
        "Market Regulation, Technology: Positive, Nationalisation, Environmental Protection, "
        "Equality: Positive, Welfare State Expansion, Welfare State Limitation, "
        "Education Expansion, Education Limitation, Immigration: Negative, Immigration: Positive, "
        "Traditional Morality: Positive, Traditional Morality: Negative, "
        "Law and Order: Positive, Law and Order: Negative, "
        "Labour Groups: Positive, Labour Groups: Negative, "
        "Underprivileged Minority Groups\n\n"
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


def extract_label_ElecDeb60to20_relations(text: str) -> str:
    match = re.search(r"(support|attack|no_relation|no relation|no rel)", text.lower())
    return match.group(1) if match else "none"


def extract_label_MotionPolicyPreference(text: str) -> str:
    match = re.search(
        r"(Agriculture and Farmers: Positive|Centralisation: Positive|Civic Mindedness: Positive|Constitutionalism: Negative|Constitutionalism: Positive|Controlled Economy|Corporatism/Mixed Economy|Culture: Positive|Decentralisation: Positive|Direct Democracy: Positive|Economic Goals|Economic Growth: Positive|Economic Orthodoxy|Economic Planning|Education Expansion|Environmental Protection|Equality: Positive|European Union: Negative|European Union: Positive|Free Market Economy|Governative and Administrative Efficiency|Human Rights|Incentives: Positive|Internationalism: Positive|Keynesian Demand Management|Labour Groups: Positive|Law and Order: Positive|Market Regulation|Middle Class and Professional Groups|Military: Negative|Military: Positive|Multiculturalism: Negative|Multiculturalism: Positive|National Way of Life: Negative|National Way of Life: Positive|Nationalisation|Non-economic Demographic Groups|Peace|Political Authority: Party|Political Corruption|Protectionism: Negative|Technology and Infrastructure: Positive|Underprivileged Minority Groups|Welfare State Expansion|Welfare State Limitation)",
        text.lower())
    return match.group(1) if match else "none"


def extract_label_ParlVotePlus(text: str) -> str:
    match = re.search(
        r"(Centralisation: Positive|Constitutionalism: Negative|Constitutionalism: Positive|Decentralisation: Positive|Direct Democracy: Positive|Education Expansion|Education Limitation|Environmental Protection|Equality: Positive|European Union: Negative|European Union: Positive|Free Market Economy|Human Rights|Immigration: Negative|Immigration: Positive|Incentives: Positive|Labour Groups: Negative|Labour Groups: Positive|Law and Order: Negative|Law and Order: Positive|Market Regulation|Military: Negative|Military: Positive|Nationalisation|Peace|Political Authority: Party|Political Authority: Personal|Political Corruption|Technology: Positive|Traditional Morality: Negative|Traditional Morality: Positive|Underprivileged Minority Groups|Welfare State Expansion|Welfare State Limitation)",
        text.lower())
    return match.group(1) if match else "none"


def compute_metrics(gold: List[str], pred: List[str]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(gold, pred),
        "precision": precision_score(gold, pred, average="macro", zero_division=0),
        "recall": recall_score(gold, pred, average="macro", zero_division=0),
        "f1": f1_score(gold, pred, average="macro", zero_division=0),
    }


def run(args):
    df = pd.read_csv(f"data/multi_class_classification/{args.dataset}/test.csv")
    pipe = load_model(args.model)

    predictions = []
    gold_labels = df[args.label_col].tolist()
    sentences = df[args.text_col].tolist()

    extract_label = {
        'ElecDeb60to20-relations': extract_label_ElecDeb60to20_relations,
        'MotionPolicyPreference': extract_label_MotionPolicyPreference,
        'ParlVote+': extract_label_ParlVotePlus,
    }

    for s in tqdm(sentences, desc="Classifying"):
        prompt = build_prompt(args.dataset, s)
        output = pipe(prompt)[0]["generated_text"]
        label = extract_label.get(args.dataset)(output)
        predictions.append(label)

    df["prediction"] = predictions
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    out_file = f"logs/{args.model}/{args.dataset}/zero_shot_multi_class_classification.csv"

    df.to_csv(out_file, index=False)
    print(f"\nSaved predictions to: {out_file}")

    metrics = compute_metrics(gold_labels, predictions)
    print("###################### RESULTS ######################")
    print(f"\nEvaluation - multiclass classification - {args.dataset}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str, choices=INSTRUCTION_PROMPT.keys(), default="ParlVote+")

    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label_value")

    args = parser.parse_args()
    run(args)
