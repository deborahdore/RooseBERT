import argparse
import os
import re
from datetime import datetime

import pandas as pd
import rootutils
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_dataset(base_path):
    dataset_dict = {}
    for file in ["train.json", "dev.json", "test.json"]:
        df = pd.read_json(os.path.join(base_path, file))
        df['text'] = df.apply(lambda row: " ".join(row.tokens), axis=1)
        df['label'] = df.apply(lambda row: integrate_ner_tags(row.tokens, row.ner_tags), axis=1)
        dataset_dict[file.split(".")[0]] = df

    return dataset_dict


# def format_dataset(examples):
#     if isinstance(examples["prompt"], list):
#         output_texts = []
#         for i in range(len(examples["prompt"])):
#             converted_sample = [
#                 {"role": "user", "content": examples["prompt"][i]},
#                 {"role": "assistant", "content": examples["completion"][i]},
#                 {"role": "user", "content": "Output: "}
#             ]
#             output_texts.append(converted_sample)
#         return {'messages': output_texts}
#         # return output_texts
#     else:
#         converted_sample = [
#             {"role": "user", "content": examples["prompt"]},
#             {"role": "assistant", "content": examples["completion"]},
#         ]
#         return {'messages': converted_sample}


def formatting_func(example):
    text = f"Sentence: {example['text']}\nArgument components: {example['label']}"
    return text


if __name__ == '__main__':
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # model_id = "google/gemma-3-4b-it"
    # model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier or path")
    args = parser.parse_args()
    model_id = args.model_id

    print(f"!! Model: {model_id}")

    # Load train/val/test split
    dataset_dict = load_dataset("./data/argument_detection/")
    train_dataset = Dataset.from_pandas(dataset_dict['train'])
    dev_dataset = Dataset.from_pandas(dataset_dict['dev'])

    # Configure Model
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0}).to(device)

    #  Load Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=transformers.TrainingArguments(
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
            output_dir=f"./logs/{model_id}",
            report_to="none"
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )
    trainer.train()

    # Save fine-tuned model
    now = datetime.now()
    trainer_filepath = f"./logs/{model_id}/argument_detection/{now.strftime('%d/%m/%y:%H:%M')}"
    trainer.save_model(trainer_filepath)

    del model, trainer
    model = AutoModelForCausalLM.from_pretrained(model_id)

    merged_model = PeftModel.from_pretrained(model, trainer_filepath)
    merged_model = merged_model.merge_and_unload()
    merged_model = merged_model.to(device)

    results = []
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    merged_model.config.pad_token_id = tokenizer.pad_token_id

    texts = dataset_dict['test']['text'].tolist()
    labels = dataset_dict['test']['label'].tolist()

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
    results_df.to_csv(f"./logs/{model_id}/results_argument_detection.csv", index=False)

    # Compute Metrics
    # preds, labels_cleaned = [], []
    # errors = 0
    #
    # for _, row in results_df.iterrows():
    #     y_true, output = row['y_true'], row['y_pred']
    #     try:
    #         if not any(label in output for label in ALLOWED_LABELS):
    #             raise ValueError("No valid label found in output.")
    #
    #         if "support" in output:
    #             preds.append(LABEL2ID_MAP["support"])
    #         elif "attack" in output:
    #             preds.append(LABEL2ID_MAP['attack'])
    #         else:
    #             preds.append(LABEL2ID_MAP['no_relation'])
    #         labels_cleaned.append(LABEL2ID_MAP[y_true])
    #
    #     except Exception:
    #         errors += 1
    #
    # print({
    #     'accuracy': accuracy_score(labels_cleaned, preds),
    #     'precision': precision_score(labels_cleaned, preds, zero_division=0, avergae='macro'),
    #     'recall': recall_score(labels_cleaned, preds, zero_division=0, avergae='macro'),
    #     'f1': f1_score(labels_cleaned, preds, zero_division=0, avergae='macro'),
    #     'errors': errors
    # })
