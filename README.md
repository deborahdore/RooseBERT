<div align="center">

# RooseBERT: A New Deal For Political Language Modelling

[![Paper](http://img.shields.io/badge/paper-arxiv.2508.03250v1-B31B1B.svg)](https://arxiv.org/abs/2508.03250v1)
</div>

All pretrained models can be found in the HuggingFace repo: [ddore14](https://huggingface.co/ddore14/models)!
If you use this model, cite us:

```bibtex 
@misc{
    dore2025roosebertnewdealpolitical, 
    title={RooseBERT: A New Deal For Political Language Modelling}, 
    author={Deborah Dore and Elena Cabrio and Serena Villata}, 
    year={2025}, 
    eprint={2508.03250}, 
    archivePrefix={arXiv}, 
    primaryClass={cs.CL}, 
    url={https://arxiv.org/abs/2508.03250}, 
} 
```

## Table of Contents

- [1️⃣ Description](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#1%EF%B8%8F⃣-description)
- [2️⃣ Datasets](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#2%EF%B8%8F⃣-datasets)
- [3️⃣ Models](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#3%EF%B8%8F⃣-models)
- [4️⃣ Installation](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#4%EF%B8%8F⃣-installation)
- [5️⃣ How to Run](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#5%EF%B8%8F⃣-how-to-run)
    - [🚀Download the Corpora](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#-download-the-corpora)
    - [🚀 Prepare the Dataset](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#-prepare-the-dataset)
    - [🚀 Running Continuous Pretraining for Masked Language Modeling](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#-running-continuous-pretraining-for-masked-language-modeling)
    - [🚀 Choose a Downstream Task](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#-choose-a-downstream-task)
- [6️⃣ Extract Results](https://github.com/deborahdore/RooseBERT?tab=readme-ov-file#-extract-results)

## 1️⃣ Description

The goal of this project is to continue the pretraining of BERT on a curated dataset of political debates.
By training BERT on a domain-specific content, we aim to generate embeddings that capture the nuanced language,
rhetoric, and argumentation style unique to political discourse.
The project will investigate whether these enhanced embeddings can improve performance in downstream tasks related to
political debates such as sentiment analysis (binary), ner, argument classification and relation classification (3-class
classification).

**Objectives**:

1. _Continuous Pre-Training_: <br>
   We pretrain BERT on political debate transcripts to generate embeddings that reflect the intricate structure
   and linguistic patterns in political dialogue.
2. _Evaluation on Downstream Tasks_: <br>
   The effectiveness of these embeddings will be assessed across a variety of downstream tasks, with a focus on tasks
   relevant to the political domain.
3. _Analysis_: <br>
   By comparing the performance of RooseBERT (our pretrained BERT model) against general BERT and similar competitor
   models, we aim to prove the effectiveness of our model in this domain.

## 2️⃣ Datasets

The following datasets were used for pre-training:

* [📌UN General Debate Corpus (UNGDC)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y)
    - A comprehensive collection of United Nations General Assembly debates from 1946 to 2023.
    - Includes over 10,000 speeches from representatives of 202 countries.
    - Accompanied by [visualization and analysis tools](https://www.ungdc.bham.ac.uk) developed by the authors.
* [📌House of Commons Parliamentary Debates](https://reshare.ukdataservice.ac.uk/854292/)
    - Contains every parliamentary debates held in the House of Commons between 1979 and 2019.
* [📌Presidential Candidates Debates](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0)
    - A collection of US presidential debates spanning from 1960 to 2024.
* [📌Australian Parliament](https://zenodo.org/records/8121950)
    - Proceedings from each sitting day in the Australian Parliament from 1998 to 2022.
* [📌 EU Speech](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XPCVEI)
    - Collection of 18,403 speeches from EU leaders from 2007 to 2015
* [📌 ParlEE UK & IE Corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZY3RV7&version=2.0)
    - Contains the full-text speeches from eight legislative chambers for Ireland and the United Kingdom, covering
      2009-2019.
* [📌 Scottish Parliament](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EQ9WBE)
    - Contains 1.8 million spoken contributions for the Scottish Parliament (up to 2021/02/03).
* [📌 United Nations Security Council](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH&version=6.1)
    - A dataset of UN Security Council debates between January 1992 and December 2023.

## 3️⃣ Models

This project fine-tunes **BERT** models:

- `bert-base-cased`
- `bert-base-uncased`
- `bert-large-uncased`
- `bert-large-cased`

## 4️⃣ Installation

#### Conda Setup

```bash

# clone project
git clone https://github.com/deborahdore/RooseBERT
cd RooseBERT

# create conda environment and install dependencies
conda env create -f environment.yaml -n rooseBERT

# activate conda environment
conda activate rooseBERT
```

## 5️⃣ How to Run

### 🚀 **Download the Corpora**

Use the script located in the [scraping](script/scraping) folder to download the datasets required for continued BERT
pre-training. Each script includes instructions at the top explaining where to obtain the datasets and how to execute
the download process.

### 🚀 **Prepare the Dataset**

Use the [`prepare_training_dataset.py`](script/prepare_training_dataset.py) script to create the train/dev split from
the raw dataset. When running the script, specify the maximum sequence length for each chunk of sentences.

_💡 Hint: For optimal BERT pre-training, we use sequences of length 128 for 80% of the time, and sequences of length 512
for
the remaining 20%._

```bash

python  script/prepare_training_dataset.py

```

### 🚀 Running Continuous Pretraining for Masked Language Modeling

To continue pretraining a model using Masked Language Modeling (MLM), you can use the [run_mlm.py](src/run_mlm.py)
script adapted from the one by Hugging Face. The pretraining process consists of two phases:

1. **First phase**: Training for **120k steps** with a maximum sequence length of **128**.
2. **Second phase**: Extending the sequence length to **512** and continuing training for a total of **150k steps**.

Below is the recommended configuration, though you can modify parameters as needed. A ready-to-run script is
provided [here](sbatch/run_mlm.sh).

#### **Phase 1: Training with Sequence Length 128**

```bash

python -m torch.distributed.launch --nproc_per_node=8 \
        --master_addr=123 \
        src/run_mlm.py \
        --model_name_or_path "bert-base-cased" \
        --cache_dir "cache/bert-base-cased-batch2048-lr5e-4/" \
        --train_file "data/training/max_128/train.csv" \
        --validation_file "data/training/max_128/dev.csv" \
        --max_seq_length 128 \
        --preprocessing_num_workers 4 \
        --output_dir "logs/bert-base-cased-batch2048-lr5e-4/" \
        --do_train \
        --do_eval \
        --eval_strategy "steps" \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-4 \
        --weight_decay 0.01 \
        --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
        --max_steps 120000 \
        --warmup_steps=10000 \
        --logging_dir "logs/bert-base-cased-batch2048-lr5e-4/" \
        --logging_strategy "steps" \
        --logging_steps 500 \
        --save_strategy "steps" \
        --save_steps 20000 \
        --save_total_limit 3 \
        --seed 42 \
        --data_seed 42 \
        --fp16 \
        --local_rank 0 \
        --eval_steps 1000 \
        --dataloader_num_workers 8 \
        --run_name "bert-base-cased-batch2048-lr5e-4" \
        --deepspeed "configs/deepspeed_config.json" \
        --report_to "wandb" \
        --eval_on_start \
        --log_level "detail"
```

#### **Phase 2: Training with Sequence Length 512**

```bash

python -m torch.distributed.launch --nproc_per_node=8 \
        --master_addr=123 \
        src/run_mlm.py \
        --model_name_or_path "logs/bert-base-cased-batch2048-lr5e-4/checkpoint-120000" \
        --overwrite_output_dir  \
        --resume_from_checkpoint "logs/bert-base-cased-batch2048-lr5e-4/checkpoint-120000" \
        --cache_dir "cache/bert-base-cased-batch2048-lr5e-4/" \
        --train_file "data/training/max_512/train.csv" \
        --validation_file "data/training/max_512/dev.csv" \
        --max_seq_length 512 \
        --preprocessing_num_workers 4 \
        --output_dir "logs/bert-base-cased-batch2048-lr5e-4/" \
        --do_train \
        --do_eval \
        --eval_strategy "steps" \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-4 \
        --weight_decay 0.01 \
        --adam_beta1 0.9 --adam_beta2 0.98 --adam_epsilon 1e-6 \
        --max_steps 150000 \
        --logging_dir "logs/bert-base-cased-batch2048-lr5e-4/" \
        --logging_strategy "steps" \
        --logging_steps 500 \
        --save_strategy "steps" \
        --save_steps 20000 \
        --save_total_limit 3 \
        --seed 42 \
        --data_seed 42 \
        --fp16 \
        --local_rank 0 \
        --eval_steps 1000 \
        --dataloader_num_workers 8 \
        --run_name "bert-base-cased-batch2048-lr5e-4" \
        --deepspeed "configs/deepspeed_config.json" \
        --report_to "wandb" \
        --eval_on_start \
        --log_level "detail"
```

#### **Notes**

- The **DeepSpeed** configuration file ([deepspeed_config.json](configs/deepspeed_config.json)) is used for optimization
  along with FP16 and gradient accumulation to speed up the training.

### 🚀 Choose a Downstream Task

We evaluated BERT on various downstream tasks relevant to natural language processing and political discourse
analysis. Below is a summary of the tasks and their datasets:

- **Sentiment Analysis**: Classify political statements as positive or negative using
  the [ParlVote](https://data.mendeley.com/datasets/czjfwgs9tm/1) Dataset (binary classification).
- **Named Entity Recognition (NER)**: Identify and categorize named entities in political debates using
  the [CrossNER](https://github.com/zliucr/CrossNER)  Dataset (political debates section).
- **Argument Detection**: Detect arguments and their structures in political speeches using
  the [ElecDeb60to20](https://github.com/pierpaologoffredo/ElecDeb60to20) Dataset.
- **Relation Classification**: Identify relationships between arguments or entities using
  the [ElecDeb60to20](https://github.com/pierpaologoffredo/ElecDeb60to20) Dataset (multi-label classification).

To download all the necessary dataset use the [download.sh](download.sh) script. Then, use
the [prepare_downstream_data.py](script/prepare_downstream_data.py) script to process all the dataset. The script will
do everything for you, just launch it.

```bash

./download.sh

python script/prepare_downstream_data.py
```

To run the _Argument Detection_ or _Named Entity Recognition (NER)_ downstream tasks, use the [
`run_ner.py`](src/run_ner.py) script. To run the _Relation Classification_ and _Sentiment Analysis_ tasks, use the [
`run_classification.py`](src/run_classification.py) script.
An example of configuration files can be found in the [sbatch](sbatch) folder.

### 🚀 Extract Results

At the end of each run, the results will be available in the `RooseBERT/logs/task_name/model_name/` folder.
The [extract_results.py](script/extract_results.py) script will automatically process the results and save them in a csv
file.

```bash

python extract_results.py
```

Ig you have run the model multiple times with different seeds, use the [compute_stats.py](script/compute_stats.py)
script to extract mean and standard deviation.

```bash

python compute_stats.py
```