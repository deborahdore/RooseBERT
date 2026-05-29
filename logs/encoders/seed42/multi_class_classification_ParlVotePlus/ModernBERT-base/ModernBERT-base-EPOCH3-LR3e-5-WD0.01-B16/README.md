---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags:
- generated_from_trainer
model-index:
- name: ModernBERT-base-EPOCH3-LR3e-5-WD0.01-B16
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-base-EPOCH3-LR3e-5-WD0.01-B16

This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on an unknown dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch_fused with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 5.8.0.dev0
- Pytorch 2.11.0+cu130
- Datasets 4.4.1
- Tokenizers 0.22.2
