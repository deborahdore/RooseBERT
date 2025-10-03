# ADAPTED FROM https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py
# !/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rootutils
import torch
from datasets import load_dataset, ClassLabel
from peft import LoraConfig, TaskType, get_peft_model
from scipy.special import softmax
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, DataCollatorForTokenClassification, EvalPrediction, BitsAndBytesConfig, GemmaForTokenClassification,
    LlamaForTokenClassification,
)
from transformers.models.mistral.modular_mistral import MistralForTokenClassification
from transformers.utils import logging as hf_logging, check_min_version
from transformers.utils.versions import require_version

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

check_min_version("4.51.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

AUTOMODEL = {
    "mistralai": MistralForTokenClassification,
    "meta-llama": LlamaForTokenClassification,
    "google": GemmaForTokenClassification,
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: Optional[str] = "ner"
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, (
                "`validation_file` should have the same extension (csv or json) as `train_file`."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./cache/",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # model_revision: str = field(
    #     default="main",
    #     metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    # )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def load_and_prepare_datasets(data_args):
    """Load datasets and prepare them for training."""
    if data_args.dataset_name:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        logger.info(f"Loaded dataset {data_args.dataset_name}")
    else:
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
            "test": data_args.test_file,
        }
        extension = data_files["train"].split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        logger.info("Loaded local dataset files.")

    return raw_datasets


def flatten(xss):
    return [x for xs in xss for x in xs]


def compute_metrics(p: EvalPrediction, label_list, output_dir: str = None, step: int = None, epoch: float = None):
    predictions, labels = p
    predictions = softmax(predictions, axis=-1)
    predictions = np.argmax(predictions, axis=-1)
    preds = flatten([
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ])
    labels = flatten([
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ])

    results = {
        'step': step,
        'epoch': epoch,
        'accuracy': accuracy_score(y_true=labels, y_pred=preds),
        'precision': precision_score(y_true=labels, y_pred=preds, average="macro"),
        'recall': recall_score(y_true=labels, y_pred=preds, average="macro"),
        'f1': f1_score(y_true=labels, y_pred=preds, average="macro")
    }

    if output_dir is not None:
        csv_file = os.path.join(output_dir, "eval_results.csv")
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'accuracy', 'precision', 'recall', 'f1'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)

    return results


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    return sorted(set(label_list))


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    hf_logging.set_verbosity_info()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # Load datasets
    raw_datasets = load_and_prepare_datasets(data_args)

    column_names = raw_datasets["train"].column_names
    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    features = raw_datasets["train"].features

    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_list = sorted(set(label_list))
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_list = sorted(set(label_list))
        label_to_id = {l: i for i, l in enumerate(label_list)}

    id_to_label = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    dsdir = os.path.join(os.getenv("DSDIR"), "HuggingFace_Models")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(dsdir, tokenizer_name_or_path),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        add_prefix_space=True
    )

    # Setup 8-bit quantization config for bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
    )
    model = AUTOMODEL[model_args.model_name_or_path.split("/")[0]].from_pretrained(
        os.path.join(dsdir, model_args.model_name_or_path),
        quantization_config=bnb_config,  # Enable 8-bit loading from bitsandbytes
        torch_dtype=torch.float16,
        # device_map="auto",  # Automatically place model on available GPU(s)
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
        finetuning_task="ner"
    )

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, peft_config)
    model.train()

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            max_length=data_args.max_seq_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    if data_args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_dataset = raw_datasets['train'].shuffle(seed=training_args.seed)
    eval_dataset = raw_datasets['validation']
    test_dataset = raw_datasets['test']

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    if data_args.max_test_samples is not None:
        max_test_samples = min(len(test_dataset), data_args.max_test_samples)
        test_dataset = test_dataset.select(range(max_test_samples))

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on train dataset",
        )

        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on validation dataset",
        )

        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on test dataset",
        )

    data_collator_with_padding = DataCollatorForTokenClassification(tokenizer, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: compute_metrics(
            p,
            label_list=label_list,
            output_dir=training_args.output_dir,
            step=trainer.state.global_step,  # Get current step
            epoch=trainer.state.epoch  # Get current epoch
        ),
        processing_class=tokenizer,
        data_collator=data_collator_with_padding,
    )
    # Training
    logger.info("*** Starting Training ***")
    train_results = trainer.train()

    trainer.save_model()
    trainer.save_state()

    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)

    logger.info("*** Starting Testing ***")
    test_results = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")

    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)

    # Testing with different seeds
    file_path = os.path.join(Path(training_args.output_dir).parent.absolute(), "random_seed_testing.csv")
    write_header = not os.path.exists(file_path)

    with open(file_path, 'a') as file:
        if write_header:
            file.write('model,seed,f1_score\n')
        file.write(f"{model_args.model_name_or_path},{training_args.seed},{test_results.metrics['test_f1']}\n")

    # logger.info("*** Starting Testing on all ckpt***")
    # metrics = []
    # training_args.report_to = []
    # ckpt_dirs = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
    # for ckpt in ckpt_dirs:
    #     logger.info(f"Evaluating checkpoint: {ckpt}")
    #     model = AutoModelForTokenClassification.from_pretrained(ckpt)
    #     model.eval()
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         eval_dataset=test_dataset,
    #         compute_metrics=lambda p: compute_metrics(
    #             p,
    #             label_list=label_list,
    #             output_dir=None,
    #             step=None,
    #             epoch=None
    #         ),
    #         processing_class=tokenizer,
    #         data_collator=data_collator_with_padding,
    #     )
    #     test_results = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
    #     results = {
    #         'ckpt': str(ckpt.split("/")[-1]),
    #         **test_results.metrics
    #     }
    #     num_steps = int(ckpt.split("-")[-1])
    #     results['test_step'] = num_steps
    #
    #     metrics.append(results)
    #
    # output_file = os.path.join(training_args.output_dir, f"ckpt_results.csv")
    # pd.DataFrame(metrics).sort_values("test_step").to_csv(output_file, index=False)
    #
    # for ckpt in ckpt_dirs:
    #     print(f"Deleting checkpoint: {ckpt}")
    #     shutil.rmtree(ckpt)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "ner"}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()

        # Clean up resources
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
