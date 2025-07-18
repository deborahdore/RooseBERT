# ADAPTED FROM https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py
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
import glob
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.special import softmax
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, EvalPrediction,
)
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

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
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
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


def preprocess_function(examples, tokenizer, max_seq_length):
    """Preprocess input examples by tokenizing them."""
    result = tokenizer(
        examples["sentence"],
        padding=False,
        max_length=max_seq_length,
        truncation=True
    )

    result["label"] = examples["label"]
    return result


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


def compute_metrics(p: EvalPrediction, output_dir: str = None, step: int = None, epoch: float = None):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    average = "binary"
    if len(set(p.label_ids)) > 2:
        average = "macro"

    preds = softmax(preds, axis=-1)
    preds = np.argmax(preds, axis=-1)

    results = {
        'step': step,
        'epoch': epoch,
        'accuracy': accuracy_score(y_true=p.label_ids, y_pred=preds),
        'precision': precision_score(y_true=p.label_ids, y_pred=preds, average=average),
        'recall': recall_score(y_true=p.label_ids, y_pred=preds, average=average),
        'f1': f1_score(y_true=p.label_ids, y_pred=preds, average=average)
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

    # Rename label column name to "label"
    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Rename text column name to "sentence"
    if data_args.text_column_name is not None and data_args.text_column_name != "sentence":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.text_column_name, "sentence")

    # Get Labels
    label_list = []
    for split in raw_datasets.keys():
        label_list.extend(raw_datasets[split].unique("label"))
    labels = sorted(list(set(label_list)))
    num_labels = len(labels)
    if num_labels <= 1:
        raise ValueError("You need more than one label to do classification.")
    label_to_id = {str(v): i for i, v in enumerate(labels)}
    id_to_label = {i: str(v) for i, v in enumerate(labels)}

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        label2id=label_to_id,
        id2label=id_to_label,
    )
    config.problem_type = "single_label_classification"

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        add_prefix_space=True if tokenizer_name_or_path.startswith("deberta") else False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.train()

    # Preprocess datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        raw_datasets = raw_datasets.map(
            lambda examples: preprocess_function(examples, tokenizer, max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on dataset",
        )
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

    # Initialize Trainer
    data_collator_with_padding = DataCollatorWithPadding(tokenizer, padding=True)
    class_weights = None

    if num_labels > 2:
        weights_list = []
        for l in sorted(labels):
            weights_list.append(train_dataset['label'].count(l))
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        counts = torch.tensor(weights_list, dtype=torch.float, device=device)
        # Inverse frequency & normalize
        weights = 1.0 / counts
        class_weights = weights * len(counts) / weights.sum()

    def compute_loss_func(outputs, labels, num_items_in_batch=None):
        loss = F.cross_entropy(input=outputs['logits'].view(-1, num_labels),
                               target=labels.view(-1),
                               weight=class_weights)
        return loss

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: compute_metrics(
            p,
            training_args.output_dir,
            trainer.state.global_step,  # Get current step
            trainer.state.epoch  # Get current epoch
        ),
        compute_loss_func=compute_loss_func if class_weights is not None else None,
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
    # file_path = os.path.join(Path(training_args.output_dir).parent.absolute(), "random_seed_testing.csv")
    # write_header = not os.path.exists(file_path)
    #
    # with open(file_path, 'a') as file:
    #     if write_header:
    #         file.write('model,seed,f1_score\n')
    #     file.write(f"{model_args.model_name_or_path},{training_args.seed},{test_results.metrics['test_f1']}\n")

    logger.info("*** Starting Testing on all ckpt***")
    metrics = []
    training_args.report_to = []
    ckpt_dirs = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")))
    for ckpt in ckpt_dirs:
        logger.info(f"Evaluating checkpoint: {ckpt}")
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        model.eval()
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=lambda p: compute_metrics(
                p,
                None,
                None,
                None),
            compute_loss_func=compute_loss_func if class_weights is not None else None,
            processing_class=tokenizer,
            data_collator=data_collator_with_padding,
        )
        test_results = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        results = {
            'ckpt': str(ckpt.split("/")[-1]),
            **test_results.metrics
        }
        num_steps = int(ckpt.split("-")[-1])
        results['test_step'] = num_steps

        metrics.append(results)

    output_file = os.path.join(training_args.output_dir, f"ckpt_results.csv")
    pd.DataFrame(metrics).sort_values("test_step").to_csv(output_file, index=False)

    for ckpt in ckpt_dirs:
        print(f"Deleting checkpoint: {ckpt}")
        shutil.rmtree(ckpt)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
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
