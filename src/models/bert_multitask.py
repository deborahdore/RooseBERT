from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import ModelOutput


@dataclass
class MultiTaskOutput(ModelOutput):
    """
    Output for multi-task learning with MLM + SCP + ACM
    """
    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    scp_loss: Optional[torch.FloatTensor] = None
    acm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    scp_logits: torch.FloatTensor = None
    acm_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForMultiTaskPretraining(BertPreTrainedModel):
    """
    BERT model with three heads:
    1. Masked Language Modeling (MLM)
    2. Speaker Change Prediction (SCP) - binary classification (same_speaker)
    3. Argument Continuity Modeling (ACM) - binary classification (same_debate)
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        # MLM head (standard)
        self.cls = nn.Linear(config.hidden_size, config.vocab_size)

        # SCP head - binary classification (same speaker or not)
        self.scp_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)  # Binary: same_speaker or different
        )

        # ACM head - binary classification (same debate or not)
        self.acm_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)  # Binary: same_debate or different
        )

        # Loss weights (you can tune these)
        self.mlm_weight = 1.0
        self.scp_weight = 0.3
        self.acm_weight = 0.3

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,  # MLM labels
            scp_labels: Optional[torch.Tensor] = None,  # Speaker change labels
            acm_labels: Optional[torch.Tensor] = None,  # Argument continuity labels
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs[1]  # [batch_size, hidden_size] - [CLS] token

        # --- MLM Head ---
        mlm_logits = self.cls(sequence_output)

        # --- SCP Head (uses [CLS] token) ---
        scp_logits = self.scp_classifier(pooled_output)

        # --- ACM Head (uses [CLS] token) ---
        acm_logits = self.acm_classifier(pooled_output)

        # --- Compute Losses ---
        total_loss = None
        mlm_loss = None
        scp_loss = None
        acm_loss = None

        if labels is not None:
            # MLM Loss (only on masked tokens)
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = self.mlm_weight * mlm_loss

        if scp_labels is not None:
            # SCP Loss (binary classification)
            loss_fct = nn.CrossEntropyLoss()
            scp_loss = loss_fct(scp_logits.view(-1, 2), scp_labels.view(-1))
            total_loss = total_loss + self.scp_weight * scp_loss if total_loss is not None else self.scp_weight * scp_loss

        if acm_labels is not None:
            # ACM Loss (binary classification)
            loss_fct = nn.CrossEntropyLoss()
            acm_loss = loss_fct(acm_logits.view(-1, 2), acm_labels.view(-1))
            total_loss = total_loss + self.acm_weight * acm_loss if total_loss is not None else self.acm_weight * acm_loss

        if not return_dict:
            output = (mlm_logits, scp_logits, acm_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskOutput(
            loss=total_loss,
            mlm_loss=mlm_loss,
            scp_loss=scp_loss,
            acm_loss=acm_loss,
            mlm_logits=mlm_logits,
            scp_logits=scp_logits,
            acm_logits=acm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
