"""
Training and evaluation utilities.

This module provides utility functions for data collation, memory encoding,
and other training-related operations.
"""

import torch
from tqdm import tqdm
from typing import List, Dict

from core.memory import MemoryEncoder, KeyValueMemoryStore


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for batching dataset items.

    Args:
        batch: List of dataset items

    Returns:
        Batched tensors
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([
            item['attention_mask'] for item in batch
        ]),
        'token_type_ids': torch.stack([
            item['token_type_ids'] for item in batch
        ]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }


def encode_paq_to_memory(
    memory_encoder: MemoryEncoder,
    memory_store: KeyValueMemoryStore,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> None:
    """
    Encode PAQ dataset into memory store.
    
    Args:
        memory_encoder: Memory encoder model
        memory_store: Key-value memory store
        dataloader: DataLoader for PAQ dataset
        device: Device to run encoding on
    """
    memory_encoder.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding PAQ to Memory"):
            questions = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            keys, values = memory_encoder(questions, attention_masks)
            
            for i in range(keys.size(0)):
                memory_store.add_entry(keys[i], values[i])
    
    print(f"Encoded {memory_store.size()} entries to memory store")