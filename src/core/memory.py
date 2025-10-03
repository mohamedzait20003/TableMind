"""
Core memory components for the DoT model.

This module implements the memory encoder and key-value store components
that provide external memory capabilities to the DoT architecture.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import T5EncoderModel


class MemoryEncoder(nn.Module):
    """
    Encodes input sequences into key-value pairs for memory storage.
    
    Uses a T5 encoder to create contextualized representations, then projects
    them into separate key and value spaces for efficient retrieval.
    """
    
    def __init__(
        self, 
        model_name: str = 't5-base', 
        hidden_size: int = 768, 
        proj_dim: int = 256
    ):
        """
        Initialize the memory encoder.
        
        Args:
            model_name: HuggingFace model identifier for T5 encoder
            hidden_size: Hidden dimension of the T5 model
            proj_dim: Projection dimension for keys and values
        """
        super(MemoryEncoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
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
        # Pool sequence representations
        pooled = outputs.last_hidden_state.mean(dim=1)
        keys = self.key_projection(pooled)
        values = self.value_projection(pooled)
        return keys, values


class KeyValueMemoryStore:
    """
    Simple key-value memory store for efficient retrieval.
    
    Stores key-value pairs on CPU to manage GPU memory usage and provides
    similarity-based retrieval functionality.
    """
    
    def __init__(self):
        """Initialize empty memory store."""
        self.keys = []
        self.values = []

    def add_entry(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Add a key-value pair to the memory store.
        
        Args:
            key: Key tensor to store
            value: Value tensor to store
        """
        # Store on CPU to avoid GPU memory growth across runs
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())

    def retrieve(
        self, 
        query_keys: torch.Tensor, 
        top_k: int = 1, 
        device: str = "cuda"
    ) -> Optional[torch.Tensor]:
        """
        Retrieve values based on key similarity.
        
        Args:
            query_keys: Query keys of shape (batch_size, proj_dim)
            top_k: Number of top similar entries to retrieve
            device: Device to perform computations on
            
        Returns:
            Retrieved values of shape (batch_size, proj_dim) or None if empty
        """
        if len(self.keys) == 0:
            return None

        store_keys = torch.stack(self.keys).to(device)
        store_values = torch.stack(self.values).to(device)
        
        # Compute similarity scores
        sim = torch.matmul(query_keys, store_keys.t())
        topk_sim, topk_idx = torch.topk(sim, k=top_k, dim=1)
        
        retrieved_values = []
        for i in range(query_keys.size(0)):
            idx = topk_idx[i]
            retrieved = store_values[idx]
            # Aggregate multiple retrieved values
            aggregated = retrieved.mean(dim=0)
            retrieved_values.append(aggregated)

        return torch.stack(retrieved_values)

    def size(self) -> int:
        """Return the number of stored entries."""
        return len(self.keys)

    def clear(self) -> None:
        """Clear all stored entries."""
        self.keys.clear()
        self.values.clear()