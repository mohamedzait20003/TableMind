"""
Memory encoder for the DoT model.

A T5 encoder whose pooled output is projected into separate key and
value spaces for the external key-value memory.
"""

import torch
import torch.nn as nn
from typing import Tuple
from transformers import T5EncoderModel


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the sequence dimension, ignoring padding positions."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class MemoryEncoder(nn.Module):
    """
    Encodes input sequences into key-value pairs for memory storage.

    Uses a T5 encoder to create contextualized representations, then projects
    them into separate key and value spaces for efficient retrieval.
    """

    def __init__(self, model_name: str = 't5-base', proj_dim: int = 256):
        """
        Initialize the memory encoder.

        Args:
            model_name: HuggingFace model identifier for T5 encoder
            proj_dim: Projection dimension for keys and values
        """
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.d_model
        self.key_projection = nn.Linear(hidden_size, proj_dim)
        self.value_projection = nn.Linear(hidden_size, proj_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input sequences into key-value pairs.

        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Tuple of (keys, values) each of shape (batch_size, proj_dim)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Pool sequence representations over non-padding positions
        pooled = masked_mean(outputs.last_hidden_state, attention_mask)
        keys = self.key_projection(pooled)
        values = self.value_projection(pooled)
        return keys, values
