import argparse
import os
import re
from datetime import datetime

import pandas as pd
import rootutils
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from trl import SFTTrainer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


def create_model(args):
    # Model configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map={"": 0}).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    return model


def integrate_ner_tags(tokens, tags):
    """
    Combine tokens and BIO tags into a single string with inline entity markup.
    """

    if len(tokens) != len(tags):
        raise ValueError("tokens and tags must have the same length.")

    result = []
    open_tag = None

    for token, tag in zip(tokens, tags):
        # Handle Outside tag (O)
        if tag == "O":
            if open_tag:
                result.append(f" </{open_tag}> ")
                open_tag = None
            result.append(token)
            continue

        # Extract label type (e.g., Claim, Premise)
        prefix, label = tag.split("-", 1)

        if prefix == "B":
            # If a tag is already open, close it before opening a new one
            if open_tag:
                result.append(f" </{open_tag}> ")
            result.append(f" <{label}> {token} ")
            open_tag = label

        elif prefix == "I":
            # Continue the current span
            if open_tag == label:
                result.append(f" {token} ")
            else:
                # Handle misaligned tags (shouldn't happen in clean BIO data)
                if open_tag:
                    result.append(f" </{open_tag}> ")
                result.append(f" <{label}> {token} ")
                open_tag = label

    # Close any unclosed tag at the end
    if open_tag:
        result.append(f" </{open_tag}> ")

    # Merge into final string
    final_text = " ".join(result)

    # Fix spacing before punctuation (optional cleanup)
    final_text = (
        final_text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" !", "!")
        .replace(" ?", "?")
        .replace(" ;", ";")
        .replace(" :", ":")
    )
    return re.sub(r' +', ' ', final_text).strip()


def convert(df_json):
    df_json['text'] = df_json.apply(lambda row: " ".join(row.tokens), axis=1)
    df_json['label'] = df_json.apply(lambda row: integrate_ner_tags(row.tokens, row.ner_tags), axis=1)
    return df_json


def run(args):
    print("Chosen Model:", args.model)
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    output_dir = f"logs/{args.model}/{args.dataset}"
    output_file = f"{output_dir}/fine_tuning_sequence_labelling.csv"

    train_df = convert(pd.read_json(f"data/sequence_labelling/{args.dataset}/train.csv"))
    dev_df = convert(pd.read_json(f"data/sequence_labelling/{args.dataset}/dev.csv"))
    test_df = convert(pd.read_json(f"data/sequence_labelling/{args.dataset}/test.csv"))

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(args)
    model.config.pad_token_id = tokenizer.pad_token_id

    # def formatting_func(example):
    #     return (
    #         "Task: Identify argumentative components in the sentence.\n"
    #         "Tag each span using <claim>...</claim> and <premise>...</premise>.\n"
    #         "Do not add or remove text.\n\n"
    #         f"Sentence:\n{example['text']}\n\n"
    #         f"Output:\n{example['label']}"
    #     )
    def preprocess(example):
        prompt = (
            "Task: Identify argumentative components in the sentence.\n"
            "Tag each span using <claim>...</claim> and <premise>...</premise>.\n"
            "Do not add or remove text.\n\n"
            f"Sentence:\n{example['text']}\n\n"
            f"Output:\n"
        )

        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        label_enc = tokenizer(
            example["label"],
            add_special_tokens=False,
        )

        input_ids = (
                prompt_enc["input_ids"]
                + label_enc["input_ids"]
                + [tokenizer.eos_token_id]
        )

        labels = (
                [-100] * len(prompt_enc["input_ids"])
                + label_enc["input_ids"]
                + [tokenizer.eos_token_id]
        )

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
    )

    dev_dataset = dev_dataset.map(
        preprocess,
        remove_columns=dev_dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            warmup_ratio=0.03,
            num_train_epochs=3,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            gradient_checkpointing=True,
            output_dir=output_file,
            report_to="none"
        ),
        peft_config=LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
        # formatting_func=formatting_func,
    )

    trainer.train()

    # Save fine-tuned model
    now = datetime.now()
    trainer_filepath = f"{output_file}/{now.strftime('%d/%m/%y:%H:%M')}"
    trainer.save_model(trainer_filepath)

    del model, trainer
    model = AutoModelForCausalLM.from_pretrained(args.model)
    merged_model = PeftModel.from_pretrained(model, trainer_filepath)
    merged_model = merged_model.merge_and_unload()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    merged_model = merged_model.to(device)

    results = []
    merged_model.config.pad_token_id = tokenizer.pad_token_id

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = (
            "Task: Identify argumentative components in the sentence.\n"
            "Tag each span using <claim>...</claim> and <premise>...</premise>.\n"
            "Do not add or remove text.\n\n"
            f"Sentence:\n{row['text']}\n\n"
            f"Output:\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = merged_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        prediction = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "text": row['text'],
            "label": row['label'],
            "prediction": prediction,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str, choices=['ElecDeb60to20-components', 'ArgUNSC'], default="ArgUNSC")

    args = parser.parse_args()
    run(args)
