import argparse
import math

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def compute_pseudo_perplexity(
        model,
        tokenizer,
        texts,
        device,
        max_length=512,
):
    model.eval()
    model.to(device)

    total_log_likelihood = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL"):
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )

            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            seq_len = input_ids.size(1)

            for i in range(1, seq_len - 1):  # skip [CLS] and [SEP]
                masked_input = input_ids.clone()
                masked_input[0, i] = tokenizer.mask_token_id

                outputs = model(
                    input_ids=masked_input,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits
                log_probs = torch.log_softmax(logits[0, i], dim=-1)

                target_id = input_ids[0, i]
                total_log_likelihood += log_probs[target_id].item()
                total_tokens += 1

    ppl = math.exp(-total_log_likelihood / total_tokens)
    return ppl


def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)

    texts = load_texts(args.test_file)

    ppl = compute_pseudo_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        max_length=args.max_length,
    )

    print(f"\nPseudo-perplexity: {ppl:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--test_file", type=str, default="data/training/max_512/dev.csv")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()
    main(args)
