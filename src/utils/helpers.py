"""
General helpers shared by the notebooks.

Configuration loading, seeding, device selection, WikiSQL batch collation
and answer-level (exact match) evaluation.
"""

import os
import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Any, List, Dict, Optional, Tuple


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base` (in place) and return it."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config. A top-level `base:` key names another config file
    (relative to this one) that is loaded first and overridden.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    base = config.pop('base', None)
    if base is None:
        return config
    base_path = os.path.join(os.path.dirname(config_path), base)
    return _deep_update(load_config(base_path), config)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> str:
    """Map 'auto' to cuda when available, else cpu."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching WikiSQL items.

    Args:
        batch: List of dataset items

    Returns:
        Batched tensors, plus the answer strings under 'answers'
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
        'answers': [item.get('answer', '') for item in batch],
    }


def exact_match(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_new_tokens: int = 32,
    max_batches: Optional[int] = None
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Generate answers and compare them to the executed SQL answers.

    Args:
        model: DoT model with a `generate` method
        dataloader: WikiSQL loader built with `collate_fn`
        device: Device to run generation on
        max_new_tokens: Generation length limit
        max_batches: Stop after this many batches (None = all)

    Returns:
        (exact-match accuracy, list of {'prediction', 'answer'} records)
    """
    model.eval()
    records = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        predictions = model.generate(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
            max_new_tokens=max_new_tokens,
        )
        for prediction, answer in zip(predictions, batch['answers']):
            records.append({'prediction': prediction, 'answer': answer})

    # Ignore case and whitespace: TAPAS splits punctuation ("guard - forward")
    def normalize(text: str) -> str:
        return ''.join(text.lower().split())

    correct = sum(
        normalize(r['prediction']) == normalize(r['answer']) for r in records
    )
    return (correct / len(records) if records else 0.0), records
