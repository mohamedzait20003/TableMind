"""
Core model components for the DoT architecture.

This module implements the main transformer components including the pruning
transformer, task transformer, and the integrated DoT model.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import (
    TapasForQuestionAnswering, 
    TapasTokenizer, 
    T5ForConditionalGeneration, 
    T5Tokenizer
)

from .memory import MemoryEncoder, KeyValueMemoryStore


class PruningTransformer(nn.Module):
    """
    Transformer model for pruning/scoring input candidates.
    
    Uses TAPAS model to score table-question pairs and select top-k
    candidates for further processing.
    """
    
    def __init__(self, model_name: str = 'google/tapas-large-finetuned-wtq'):
        """
        Initialize the pruning transformer.
        
        Args:
            model_name: HuggingFace model identifier for TAPAS model
        """
        super().__init__()
        self.tapas = TapasForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Score input candidates for pruning.
        
        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type ids for table structure
            
        Returns:
            Scores for each token position
        """
        outputs = self.tapas(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return outputs.logits


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
        proj_dim: int = 256
    ):
        """
        Initialize the task transformer.
        
        Args:
            memory_store: Key-value memory store for retrieval
            memory_encoder: Memory encoder for query generation
            model_name: HuggingFace model identifier for T5 model
            proj_dim: Projection dimension for memory integration
        """
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.memory_store = memory_store
        self.memory_encoder = memory_encoder

        # Projection layers for memory integration
        self.query_projection = nn.Linear(self.t5.config.d_model, proj_dim)
        self.memory_projector = nn.Linear(proj_dim, self.t5.config.d_model)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        table_embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with memory-augmented generation.
        
        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            table_embeddings: Table embeddings from memory encoder
            labels: Target labels for training
            
        Returns:
            Model outputs with loss and logits
        """
        # Get text encoder hidden states
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        text_encoder_hidden = encoder_outputs.last_hidden_state

        batch_size, seq_len, _ = text_encoder_hidden.size()
        
        # Integrate table embeddings
        table_embeddings = table_embeddings.unsqueeze(1).expand(
            -1, seq_len, -1
        )
        table_embeddings = self.memory_projector(table_embeddings)
        combined_hidden = text_encoder_hidden + table_embeddings

        # Memory retrieval
        pooled = combined_hidden.mean(dim=1)
        query_vec = self.query_projection(pooled)
        retrieved_memory = self.memory_store.retrieve(
            query_vec, top_k=1, device=input_ids.device
        )
        
        if retrieved_memory is not None:
            retrieved_embed = self.memory_projector(retrieved_memory)
            combined_hidden = combined_hidden + retrieved_embed.unsqueeze(1)

        # Generate output
        outputs = self.t5(
            encoder_outputs=(combined_hidden,),
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs


class DoTModel(nn.Module):
    """
    Complete DoT (Differentiable Optimized Transformer) model.
    
    Integrates pruning transformer and task transformer with memory
    for end-to-end table question answering.
    """
    
    def __init__(
        self, 
        memory_store: KeyValueMemoryStore, 
        memory_encoder: MemoryEncoder, 
        k: int = 128,
        pruning_model: str = 'google/tapas-large-finetuned-wtq',
        task_model: str = 't5-base'
    ):
        """
        Initialize the complete DoT model.
        
        Args:
            memory_store: Key-value memory store
            memory_encoder: Memory encoder
            k: Number of top candidates to select
            pruning_model: Model name for pruning transformer
            task_model: Model name for task transformer
        """
        super().__init__()
        self.k = k
        self.pruner = PruningTransformer(pruning_model)
        self.task_model = TaskTransformer(
            memory_store, memory_encoder, task_model
        )

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the complete DoT pipeline.
        
        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type ids for table structure
            labels: Target labels for training
            
        Returns:
            Model outputs from task transformer
        """
        # Pruning step
        scores = self.pruner(input_ids, attention_mask, token_type_ids)
        topk_scores, topk_indices = scores.topk(self.k, dim=1)

        # Select top-k candidates
        selected_input_ids = torch.gather(
            input_ids, dim=1, index=topk_indices
        )
        selected_attention_mask = torch.gather(
            attention_mask, dim=1, index=topk_indices
        )

        # Get table embeddings from memory encoder
        table_embeddings = self.task_model.memory_encoder(
            selected_input_ids, selected_attention_mask
        )[0]

        # Task transformer forward pass
        outputs = self.task_model(
            input_ids=selected_input_ids,
            attention_mask=selected_attention_mask,
            table_embeddings=table_embeddings,
            labels=labels,
        )
        return outputs