from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForMultiTaskPretraining(DataCollatorForLanguageModeling):
    """
    Data collator that handles MLM + SCP + ACM tasks
    Extends the standard DataCollatorForLanguageModeling
    """

    def __call__(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # First, handle MLM masking using parent class
        batch = super().__call__(examples)

        # Now add SCP and ACM labels
        if "scp_labels" in examples[0]:
            scp_labels = torch.tensor([ex["scp_labels"] for ex in examples], dtype=torch.long)
            batch["scp_labels"] = scp_labels

        if "acm_labels" in examples[0]:
            acm_labels = torch.tensor([ex["acm_labels"] for ex in examples], dtype=torch.long)
            batch["acm_labels"] = acm_labels

        return batch
