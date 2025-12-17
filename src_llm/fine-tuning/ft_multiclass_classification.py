import argparse
import os
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


def run(args):
    print("Chosen Model:", args.model)
    os.makedirs(f"logs/{args.model}/{args.dataset}", exist_ok=True)
    output_dir = f"logs/{args.model}/{args.dataset}"

    train_df = pd.read_csv(f"data/binary_classification/{args.dataset}/train.csv")
    dev_df = pd.read_csv(f"data/binary_classification/{args.dataset}/dev.csv")
    test_df = pd.read_csv(f"data/binary_classification/{args.dataset}/test.csv")

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_model(args)
    model.config.pad_token_id = tokenizer.pad_token_id
    label_str = ", ".join(sorted(set(train_df[args.label_col].tolist())))

    def preprocess(example):
        prompt = (
            "Task: Classify the sentence.\n"
            f"Choose exactly one label from: {label_str}.\n"
            "Only output the label.\n\n"
            f"Sentence:\n{example[args.text_col]}\n\n"
            "Output:\n"
        )

        prompt_enc = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
        )

        label_enc = tokenizer(
            example[args.label_col],
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
            gradient_checkpointing=True,
            output_dir=output_dir,
            report_to="none",
            bf16=True,
            fp16=False,
        ),
        peft_config=LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
    )

    trainer.train()

    # Save fine-tuned model
    now = datetime.now()
    trainer_filepath = f"{output_dir}/{now.strftime('%d/%m/%y:%H:%M')}"
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
            "Task: Classify the sentence.\n"
            f"Choose exactly one label from: {label_str}.\n"
            "Only output the label.\n\n"
            f"Sentence:\n{row[args.text_col]}\n\n"
            "Output:\n"
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
                max_new_tokens=50,
                do_sample=False,
            )

        prediction = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        results.append({
            "text": row[args.text_col],
            "label": row[args.label_col],
            "prediction": prediction,
        })

    results_df = pd.DataFrame(results)
    out_file = f"{output_dir}/fine_tuning_multi_class_classification.csv"

    results_df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str,
                        choices=['ArgUNSC', 'ElecDeb60to20-relations', 'MotionPolicyPreference', 'ParlVote+'],
                        default="ParlVote")

    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label_value")

    args = parser.parse_args()
    run(args)
