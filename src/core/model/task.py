"""
Task transformer for the DoT model.

A T5 encoder-decoder whose encoder states are augmented with table
embeddings and values retrieved from the key-value memory.
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from ..memory import MemoryEncoder, KeyValueMemoryStore, masked_mean


class TaskTransformer(nn.Module):
    """
    Main task transformer with memory integration.

    Integrates retrieved memory with T5 encoder-decoder for conditional
    generation on table-question answering tasks.
    """

    def __init__(
        self,
        memory_store: KeyValueMemoryStore,
        memory_encoder: MemoryEncoder,
        model_name: str = 't5-base',
        memory_top_k: int = 8
    ):
        """
        Initialize the task transformer.

        Args:
            memory_store: Key-value memory store for retrieval
            memory_encoder: Memory encoder for table embeddings
            model_name: HuggingFace model identifier for T5 model
            memory_top_k: Number of memory entries mixed per query
        """
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.memory_store = memory_store
        self.memory_encoder = memory_encoder
        self.memory_top_k = memory_top_k

        # Projection layers for memory integration; the projection width
        # always matches the memory encoder's key/value width
        proj_dim = memory_encoder.key_projection.out_features
        self.query_projection = nn.Linear(self.t5.config.d_model, proj_dim)
        self.memory_projector = nn.Linear(proj_dim, self.t5.config.d_model)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Memory-augmented encoding.

        Args:
            input_ids: T5 token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_scale: Per-token multiplier for the input embeddings
                (pruning scores), shape (batch_size, seq_len)

        Returns:
            Encoder hidden states of shape (batch_size, seq_len, d_model)
        """
        inputs_embeds = self.t5.get_input_embeddings()(input_ids)
        if token_scale is not None:
            inputs_embeds = inputs_embeds * token_scale.unsqueeze(-1)

        text_encoder_hidden = self.t5.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Integrate table embeddings
        table_embeddings, _ = self.memory_encoder(input_ids, attention_mask)
        combined_hidden = (
            text_encoder_hidden
            + self.memory_projector(table_embeddings).unsqueeze(1)
        )

        # Memory retrieval
        pooled = masked_mean(combined_hidden, attention_mask)
        query_vec = self.query_projection(pooled)
        retrieved_memory = self.memory_store.retrieve(
            query_vec, top_k=self.memory_top_k, device=input_ids.device
        )

        if retrieved_memory is not None:
            retrieved_embed = self.memory_projector(retrieved_memory)
            combined_hidden = combined_hidden + retrieved_embed.unsqueeze(1)

        return combined_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_scale: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with memory-augmented generation.

        Args:
            input_ids: T5 token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_scale: Per-token embedding multiplier (pruning scores)
            labels: Target labels for training (-100 = ignored)

        Returns:
            Model outputs with loss and logits
        """
        combined_hidden = self.encode(input_ids, attention_mask, token_scale)
        return self.t5(
            encoder_outputs=(combined_hidden,),
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
