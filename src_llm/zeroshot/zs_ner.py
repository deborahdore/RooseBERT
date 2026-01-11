import argparse
import logging
import os
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
        "Task: identify the entities in the sentence.\n\n"
        "Possible categories are:\n"
        "CC (coordinating conjunction, e.g. 'and', 'but'), "
        "CD (cardinal number, e.g. 'three'), "
        "DT (determiner, e.g. 'the'), "
        "EX (existential 'there'), "
        "FW (foreign word, e.g. 'ad hoc'), "
        "IN (preposition or subordinating conjunction, e.g. 'because'), "
        "JJ (adjective, e.g. 'strong'), "
        "JJR (comparative adjective, e.g. 'stronger'), "
        "JJS (superlative adjective, e.g. 'strongest'), "
        "LS (list marker, e.g. '1.'), "
        "MD (modal verb, e.g. 'must'), "
        "NN (singular noun, e.g. 'argument'), "
        "NNP (proper noun, e.g. 'Jennifer'), "
        "NNPS (plural proper noun, e.g. 'Americans'), "
        "NNS (plural noun, e.g. 'arguments'), "
        "PDT (predeterminer, e.g. 'all'), "
        "POS (possessive ending, e.g. \"'s\"), "
        "PRP (personal pronoun, e.g. 'she'), "
        "RB (adverb, e.g. 'clearly'), "
        "RBR (comparative adverb, e.g. 'more'), "
        "RBS (superlative adverb, e.g. 'most'), "
        "RP (particle, e.g. 'up' in 'give up'), "
        "SYM (symbol, e.g. '$'), "
        "TO (infinitive marker 'to'), "
        "UH (interjection, e.g. 'oh'), "
        "VB (base verb, e.g. 'argue'), "
        "VBD (past verb, e.g. 'argued'), "
        "VBG (gerund, e.g. 'arguing'), "
        "VBN (past participle, e.g. 'supported'), "
        "VBP (present verb, e.g. 'believe'), "
        "VBZ (3rd person present verb, e.g. 'believes'), "
        "WDT (wh-determiner, e.g. 'which'), "
        "WP (wh-pronoun, e.g. 'who'), "
        ", (comma), "
        ": (colon), "
        ". (dot), "
        "WRB (wh-adverb, e.g. 'why').\n\n"
        "Rewrite the entire sentence exactly as given.\n"
        "Surround each identified entity span with <tag>...</tag>.\n"
        "Do not add, remove, or change any words.\n\n"
        "Output the fully tagged sentence only.\n\n"
        "Sentence: {sentence}\n"
        "Output:"
    )
}

import re


def transform_into_tags(original_sentence: str, predicted_text: str, length: int):
    """
    Convert text with <TAG>...</TAG> into flat token-level tags
    aligned to the original sentence tokens (no BIO scheme).
    """
    tokens = original_sentence.strip().split()
    tags = ["O"] * len(tokens)

    pattern = re.compile(
        r"<(,|\.|:|CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|LS|MD|NN|NNP|NNPS|NNS|PDT|POS|PRP|RB|RBR|RBS|RP|SYM|TO|UH|VB|VBD|VBG|VBN|VBP|VBZ|WDT|WP|WRB)>(.*?)</\1>",
        re.DOTALL | re.IGNORECASE
    )

    spans = pattern.findall(predicted_text)

    for tag, span_text in spans:
        tag = tag.upper()
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

        for pos in matched_indices:
            tags[pos] = tag

    return tags[:length]


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
    parser.add_argument("--dataset", type=str, default="nerex", choices=["nerex"])
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    args = parser.parse_args()

    # Load dataset
    dataset = load_data(f"./data/ner/{args.dataset}/test.json")

    # Build prompt
    dataset["prompt"] = dataset["text"].apply(lambda x: PROMPT_TEMPLATES["instructions"].format(sentence=x))

    # Prepare output directory
    model_name = args.model.split("/")[-1]
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    out_file = f"logs/{args.model}/{args.dataset}/zero_shot_ner.csv"

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
