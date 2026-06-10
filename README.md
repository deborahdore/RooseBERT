<div align="center">

# RooseBERT: A New Deal For Political Language Modelling
[![arXiv](https://img.shields.io/badge/arXiv-2508.03250-b31b1b.svg)](https://arxiv.org/abs/2508.03250)
</div>

Our models are available on HuggingFace, in the [RooseBERT's collection](https://huggingface.co/collections/MARIANNE-INRIA/roosebert). If you use them, cite
us:

```bibtex
@misc{dore2025roosebertnewdealpolitical,
    title = {RooseBERT: A New Deal For Political Language Modelling},
    author = {Deborah Dore and Elena Cabrio and Serena Villata},
    year = {2025},
    eprint = {2508.03250},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2508.03250},
}
```

<!-- TOC -->
* [RooseBERT: A New Deal For Political Language Modelling](#roosebert-a-new-deal-for-political-language-modelling)
  * [1️⃣ Description](#1-description)
  * [2️⃣ Datasets](#2-datasets)
  * [3️⃣ Models](#3-models)
  * [4️⃣ Installation](#4-installation)
      * [Conda Setup](#conda-setup)
  * [5️⃣ How to Run](#5-how-to-run)
    * [🚀 **Download the Corpora**](#-download-the-corpora)
    * [🚀 Pre-Training: Continued Pre-Training (CONT)](#-pre-training-continued-pre-training-cont)
      * [**Phase 1: Training with Sequence Length 128**](#phase-1-training-with-sequence-length-128)
      * [**Phase 2: Training with Sequence Length 512**](#phase-2-training-with-sequence-length-512)
      * [**Notes**](#notes)
    * [🚀 Pre-Training: Training from Scratch (SCR)](#-pre-training-training-from-scratch-scr)
    * [🚀 Downstream Tasks](#-downstream-tasks)
    * [🚀 Extract Results](#-extract-results)
    * [Acknowledgement](#acknowledgement)
<!-- TOC -->

## 1️⃣ Description

The goal of this project is to pre-train a domain-specific language model on a curated corpus of English political
debates. By training on domain-specific content, we aim to generate embeddings that capture the nuanced language,
rhetoric, and argumentation style unique to political discourse.
The project investigates whether these enhanced embeddings improve performance on downstream tasks related to
political debates such as sentiment analysis, stance detection, argument classification and relation classification.

RooseBERT was trained using two strategies:

1. **Continued Pre-Training (CONT)**: We initialise from BERT's original weights and vocabulary and continue
   training on the political debate corpus.
2. **Training from Scratch (SCR)**: We train BERT from random initialisation using a custom WordPiece tokenizer
   built from the domain corpus. This produces a domain-specific vocabulary that encodes political terminology
   as single tokens.

Each strategy was applied in both **cased** and **uncased** variants, yielding four RooseBERT models in total.

**Objectives**:

1. _Pre-Training_: <br>
   We pre-train BERT (CONT and SCR) on political debate transcripts to generate embeddings that reflect the
   intricate structure and linguistic patterns in political dialogue.
2. _Evaluation on Downstream Tasks_: <br>
   The effectiveness of these embeddings is assessed across a variety of downstream tasks, with a focus on tasks
   relevant to the political domain.
3. _Analysis_: <br>
   By comparing the performance of RooseBERT against BERT, ModernBERT, ConfliBERT, and PoliBERTweet, we
   demonstrate the effectiveness of domain-specific pre-training for political NLP.

## 2️⃣ Datasets

The following datasets were used for pre-training:

* [📌 HOME Project Parliamentary Activity Datasets](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISX4G)
    * The HOME Project parliamentary activity datasets cover the Ghanaian Parliament (Parliament of Ghana), and the
      South African Parliament (Parliament of the Republic of South Africa).
* [📌 Australian Parliament](https://zenodo.org/records/17351233)
    * Proceedings from each sitting day in the Australian Parliament from 1998 to 2025.
* [📌 Canadian Parliament](https://openparliament.ca/debates/)
    * Scraped speeches from the Canadian Parliament.
* [📌 EU Speech](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XPCVEI)
    * Collection of 18,403 speeches from EU leaders from 2007 to 2015
* [📌 ParlEE IE Corpus](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ZY3RV7&version=2.0)
    * Contains the full-text speeches from eight legislative chambers for Ireland, covering 2009-2019.
* [📌 Parliamentary Speeches in Ireland](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6MZN76)
    * Contains parliamentary speeches in Ireland from 1919 to 2023.
* [📌 Parliamentary Speeches in New Zealand](https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/L4OAKN/LLMYON&version=1.0)
* [📌 Scottish Parliament](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EQ9WBE)
    - Contains 1.8 million spoken contributions for the Scottish Parliament (up to 2021/02/03).
* [📌House of Commons Parliamentary Debates](https://reshare.ukdataservice.ac.uk/854292/)
    - Contains every parliamentary debates held in the House of Commons in UK between 1979 and 2019.
* [📌UN General Debate Corpus (UNGDC)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0TJX8Y)
    - A comprehensive collection of United Nations General Assembly debates from 1946 to 2023.
    - Includes over 10,000 speeches from representatives of 202 countries.
    - Accompanied by [visualization and analysis tools](https://www.ungdc.bham.ac.uk) developed by the authors.
* [📌 United Nations Security Council](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KGVSYH&version=6.1)
    - A dataset of UN Security Council debates between January 1992 and December 2023.
* [📌Presidential Candidates Debates](https://www.presidency.ucsb.edu/documents/presidential-documents-archive-guidebook/presidential-campaigns-debates-and-endorsements-0)
    - A collection of US presidential debates spanning from 1960 to 2024.

## 3️⃣ Models

This project produces **RooseBERT**, a domain-specific language model for English political debates, in four variants:

| Model | Strategy | Vocab |
|---|---|---|
| [`RooseBERT-cont-cased`](https://huggingface.co/MARIANNE-INRIA/RooseBERT-cont-cased) | Continued pre-training from `bert-base-cased` | Original BERT cased vocab |
| [`RooseBERT-cont-uncased`](https://huggingface.co/MARIANNE-INRIA/RooseBERT-cont-uncased) | Continued pre-training from `bert-base-uncased` | Original BERT uncased vocab |
| [`RooseBERT-scr-cased`](https://huggingface.co/MARIANNE-INRIA/RooseBERT-scr-cased) | Trained from scratch | Custom cased WordPiece vocab |
| [`RooseBERT-scr-uncased`](https://huggingface.co/MARIANNE-INRIA/RooseBERT-scr-uncased) | Trained from scratch | Custom uncased WordPiece vocab |

Comparison baselines used in the paper: `bert-base-cased`, `bert-base-uncased`, `ModernBERT-base`,
`ConfliBERT-cont-cased`, `ConfliBERT-cont-uncased`, `ConfliBERT-scr-cased`, `ConfliBERT-scr-uncased`,
and `PoliBERTweet`.

## 4️⃣ Installation

#### Conda Setup

```bash

# clone project
git clone https://github.com/MARIANNE-INRIA/RooseBERT
cd RooseBERT

# create conda environment and install dependencies
conda env create -f environment.yaml -n rooseBERT

# activate conda environment
conda activate rooseBERT
```

## 5️⃣ How to Run

### 🚀 **Download the Corpora**

Use the [download_pretraining_data.sh](download_pretraining_data.sh) script to download and prepare the datasets
required for continued BERT pre-training.
This script will use the [`prepare_training_dataset.py`](script/prepare_training_dataset.py) script to create the
train/dev split from the raw dataset.

_💡 Hint: For optimal BERT pre-training, we use sequences of length 128 for 80% of the time, and sequences of length 512
for the remaining 20%._

```bash

python  script/prepare_training_dataset.py

```

### 🚀 Pre-Training: Continued Pre-Training (CONT)

To continue pre-training BERT using Masked Language Modeling (MLM), use the [run_mlm.py](src/run_mlm.py)
script and the [run_mlm.sh](sh/run_mlm.sh) shell script. The pre-training process consists of two phases:

1. **Phase 1**: Train for **120k steps** with a maximum sequence length of **128**.
2. **Phase 2**: Resume from the Phase 1 checkpoint and continue to a **cumulative total of 150k steps**
   (i.e., 30k additional steps) with a maximum sequence length of **512**.

Below is the recommended configuration, though you can modify parameters as needed. A ready-to-run script is
provided [here](sh/run_mlm.sh).

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
- The above example uses `bert-base-cased`; replace with `bert-base-uncased` for the uncased CONT variant.

### 🚀 Pre-Training: Training from Scratch (SCR)

To train RooseBERT from scratch with a custom domain vocabulary, use the [run_mlm_scratch.sh](sh/run_mlm_scratch.sh)
script. A custom WordPiece tokenizer must be trained first on the political debate corpus and saved to
`./tokenizer_cased/` or `./tokenizer_uncased/`.

The SCR pre-training also uses two phases:

1. **Phase 1**: Train for **200k steps** with a maximum sequence length of **128**.
2. **Phase 2**: Resume from the Phase 1 checkpoint and continue to a **cumulative total of 250k steps**
   (i.e., 50k additional steps) with a maximum sequence length of **512**.

```bash
# Edit run_mlm_scratch.sh to set TYPE="cased" or TYPE="uncased", then:
sbatch sh/run_mlm_scratch.sh
```

### 🚀 Downstream Tasks

We evaluated RooseBERT and all comparison models (BERT, ModernBERT, ConfliBERT, PoliBERTweet) on the following
downstream tasks. Below is a summary of the tasks and their datasets:

* **ParlVote** (sentence-pair, binary classification)
    * Sentiment analysis of UK Parliamentary Debates using both motion and speech text
* **HanDeSeT** (binary classification)
    * Sentiment analysis of UK Parliamentary Debates
* **ConVote** (binary classification)
    * Stance detection of US Congressional floor debates
* **AusHansard** (binary classification, cross-domain)
    * Stance detection on Australian Parliamentary Debates; used for cross-domain evaluation
* **ElecDeb60to20** — two tasks:
    * Argument Component Detection and Classification (sequence labelling) in US Presidential Debates
    * Argument Relation Prediction and Classification (sentence-pair, multi-class) in US Presidential Debates
* **ArgUNSC** — two tasks:
    * Argument Component Detection and Classification (sequence labelling) in UN Security Council debates
    * Argument Relation Prediction and Classification (sentence-pair, multi-class) in UN Security Council debates
* **ParlVote+** (multi-class classification)
    * Policy preference classification of UK Parliamentary speeches (34 policy categories)
* **NEREx** (NER / token classification)
    * Named entity recognition in US Presidential Debate transcripts (37 entity types)

To sum up:

|       **Task Type**        | **Count** |
|:--------------------------:|-----------|
|   binary classification    | 4         |
| multi-class classification | 4         |
|     sequence labelling     | 3         |

|  **Task Type**  | **Count** |
|:---------------:|-----------|
| single sentence | 5         |
|  sentence-pair  | 3         |
|       ner       | 3         |

|                       **Task Type**                       | **Count** |
|:---------------------------------------------------------:|-----------|
|                    sentiment analysis                     | 2         |
|                     stance detection                      | 2         |
|             policy preference classification              | 2         |    
|      argument component detection and classification      | 2         | 
| argument component relation prediction and classification | 2         |  
|                            NER                            | 1         |

To download all the necessary datasets use the [download_downstream_data.sh](download_downstream_data.sh) script.
Then use the [prepare_downstream_data.py](script/prepare_downstream_data.py) script to process all datasets.

```bash

./download_downstream_data.sh

python script/prepare_downstream_data.py
```

### 🚀 Extract Results

At the end of each run, the results will be available in the `RooseBERT/logs/task_name/model_name/` folder.
The [extract_results.py](script/extract_results.py) script will automatically process the results and save them in a csv
file.

```bash

python extract_results.py
```

If you have run the model multiple times with different seeds, use the [compute_stats.py](script/compute_stats.py)
script to extract mean and standard deviation.

```bash

python compute_stats.py
```

### Acknowledgement

This work has been supported by the French government, through the 3IA Cote d’Azur Investments in the project managed by
the National Research Agency (ANR) with the reference number ANR-23-IACL-0001. This project was provided with computing
AI and storage resources by GENCI at IDRIS thanks to the grant 2026-AD011016047R1 on the supercomputer Jean Zay’s A100
partition.