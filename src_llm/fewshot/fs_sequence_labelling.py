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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

from utils import flatten, load_data

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

PROMPT_TEMPLATES = {
    "instructions": (
        "Task: identify argumentative spans in the sentence.\n\n"
        "Span types:\n"
        "- <claim>: expresses a stance, opinion, or proposed policy\n"
        "- <premise>: provides justification or support for a claim\n\n"
        "Rewrite the entire sentence exactly as given.\n"
        "Surround each identified span with <claim>...</claim> or <premise>...</premise>.\n"
        "Only tag spans that clearly match one of the types.\n"
        "Do not add, remove, or change any words.\n\n"
        "Output the fully tagged sentence only.\n\n"
        "{examples}"
        "Sentence: {sentence}\n"
        "Output:"
    ),
    "ElecDeb60to20": (
        "Sentence: Yes, I voted for it, supported it.\n"
        "Output: <premise>Yes, I voted for it</premise>, <claim>supported it.</claim>\n\n"

        "Sentence: Next question here for President Clinton. Yes, ma'am, here on the front row.\n"
        "Output: Next question here for President Clinton. Yes, ma'am, here on the front row.\n\n"

        "Sentence: Not some of the military. That was the decision of the Joint Chiefs of Staff, recommended to us and agreed to by the president. That is a fact.\n"
        "Output: <claim>Not some of the military.</claim> <premise>That was the decision of the Joint Chiefs of Staff, recommended to us and agreed to by the president. That is a fact.</premise>\n\n"
    ),
    "ArgUNSC": (
        "Sentence: United Nations monitors have reported a consistent reinforcement of barricades and armed civilians on both sides.\n"
        "Output: <premise>United Nations monitors have reported a consistent reinforcement of barricades and armed civilians on both sides.</premise>\n\n"

        "Sentence: The situation is therefore now more combustible than ever.\n"
        "Output: </claim>The situation is therefore now more combustible than ever.<claim>\n\n"

        "Sentence: The Russian Federation has called for this emergency meeting of the Security Council because of the serious dangerous evolution of the situation in south-eastern Ukraine.\n"
        "Output: The Russian Federation has called for this emergency meeting of the Security Council because of the serious dangerous evolution of the situation in south-eastern Ukraine.\n\n"
    ),
}


def transform_into_tags(original_sentence: str, predicted_text: str, length: int):
    """
    Convert text with <claim>...</claim> and <premise>...</premise> into BIO tags aligned
    to the original sentence tokens.
    """
    tokens = original_sentence.strip().split()
    bio = ["O"] * len(tokens)

    pattern = re.compile(r"<(claim|premise)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
    spans = pattern.findall(predicted_text)

    for tag, span_text in spans:
        tag = tag.lower()
        span_tokens = span_text.strip().split()
        matched_indices = []

        # Greedy matching of span tokens
        for span_tok in span_tokens:
            for i, tok in enumerate(tokens):
                if i in matched_indices:
                    continue
                if tok == span_tok:
                    matched_indices.append(i)
                    break

        matched_indices.sort()
        for idx, pos in enumerate(matched_indices):
            bio[pos] = ("B-" if idx == 0 else "I-") + tag

    return bio[:length]


def load_model_and_pipeline(model_id: str):
    """Load tokenizer, model, and text generation pipeline."""

    quant = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if "gemma" in model_id:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant,
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    generator = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=False,
        max_new_tokens=512,
        repetition_penalty=1.1,
        dtype=torch.bfloat16,
        return_full_text=False,
    )

    return generator


def run_model(generator, dataset, results_file, batch_size=8):
    prompts = dataset["prompt"].tolist()
    sentences = dataset["text"].tolist()
    gold_labels = dataset["ner_tag"].apply(lambda x: x.split()).tolist()

    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_sentences = sentences[i:i + batch_size]
        batch_gold = gold_labels[i:i + batch_size]

        with torch.inference_mode():
            batch_outputs = generator(batch_prompts)

        for sent, gold, out in zip(batch_sentences, batch_gold, batch_outputs):
            outputs.append({
                "text": sent,
                "ner_tags": gold,
                "response": out[0]["generated_text"]
            })

    df = pd.DataFrame(outputs)
    df.to_csv(results_file, index=False)
    return df


def evaluate(results_df):
    preds = []
    gold = []

    for _, row in results_df.iterrows():
        bio_pred = transform_into_tags(
            original_sentence=row["text"],
            predicted_text=row["response"],
            length=len(row["ner_tags"])
        )
        preds.append(bio_pred)
        gold.append(row["ner_tags"])

    flat_preds = flatten(preds)
    flat_gold = flatten(gold)

    return {
        "accuracy": accuracy_score(flat_gold, flat_preds),
        "precision": precision_score(flat_gold, flat_preds, average="macro"),
        "recall": recall_score(flat_gold, flat_preds, average="macro"),
        "f1": f1_score(flat_gold, flat_preds, average="macro"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ElecDeb60to20",
                        choices=["ArgUNSC", "ElecDeb60to20"])
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    args = parser.parse_args()

    # Load dataset
    dataset = load_data(f"./data/sequence_labelling/{args.dataset}/test.json")

    # Build prompt
    examples = PROMPT_TEMPLATES[args.dataset].strip()
    dataset["prompt"] = dataset["text"].apply(
        lambda x: PROMPT_TEMPLATES["instructions"].format(examples=examples, sentence=x))

    # Prepare output directory
    model_name = args.model.split("/")[-1]
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    out_file = f"logs/{args.model}/{args.dataset}/few_shot_sequence_labelling.csv"

    # Run model
    generator = load_model_and_pipeline(args.model)
    results_df = run_model(generator, dataset, out_file)

    # Evaluate
    # metrics = evaluate(results_df)
    #
    # logging.info("###################### RESULTS ######################")
    # print(f"\nEvaluation - sequence labelling - {args.dataset}:")
    # for k, v in metrics.items():
    #     logging.info(f"{k.capitalize()}: {v}")
    # logging.info("######################################################")
