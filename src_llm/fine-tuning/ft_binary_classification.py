import argparse
from datetime import datetime

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, AutoTokenizer
from trl import SFTTrainer


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


ID2LABELS = {
    "AusHansard": {1: 'support', 0: 'oppose'},
    "ConVote": {1: 'support', 0: 'oppose'},
    "ParlVote": {1: 'positive', 0: 'negative'},
    "HanDeSet": {1: 'positive', 0: 'negative'},
}


def run(args):
    print("Chosen Model:", args.model)
    output_dir = f"./logs/{args.model}/binary_classification_{args.dataset}/"

    train_df = pd.read_csv(f"data/binary_classification/{args.dataset}/train.csv")
    dev_df = pd.read_csv(f"data/binary_classification/{args.dataset}/dev.csv")
    test_df = pd.read_csv(f"data/binary_classification/{args.dataset}/test.csv")

    train_df[args.label_col] = train_df[args.label_col].map(ID2LABELS[args.dataset])
    dev_df[args.label_col] = dev_df[args.label_col].map(ID2LABELS[args.dataset])
    test_df[args.label_col] = test_df[args.label_col].map(ID2LABELS[args.dataset])

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    model = create_model(args)

    def formatting_func(example):
        text = f"Sentence: {example[args.text_col]}\nOutput: {example[args.label_col]}"
        return text

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
            output_dir=output_dir,
            report_to="none"
        ),
        peft_config=LoraConfig(
            r=8,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
        formatting_func=formatting_func,
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
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    merged_model.config.pad_token_id = tokenizer.pad_token_id

    texts = test_df[args.text_col].tolist()
    labels = test_df[args.label_col].tolist()

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = merged_model.generate(
                **inputs,
                max_new_tokens=50
            )

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text, label, pred in zip(batch_texts, batch_labels, predictions):
            results.append({
                'text': text,
                'label': label,
                'prediction': pred
            })
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/results.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--dataset", type=str, choices=['AusHansard', 'HanDeSeT', 'ParlVote', 'ConVote'],
                        default="ParlVote")

    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")

    args = parser.parse_args()
    run(args)
