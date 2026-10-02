"""
General helpers shared by the notebooks.

Configuration loading, seeding, device selection and WikiSQL batch
collation.
"""

import os
import yaml
import torch
import random
import numpy as np
from typing import Any, List, Dict


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
        Batched tensors, plus answer strings ('answers') and answer value
        lists ('answer_values')
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
        'answer_values': [item.get('answer_values', []) for item in batch],
    }


def run_name(config: Dict[str, Any]) -> str:
    """
    Name of a pretraining run: experiment, memory ablation and seed, so the
    3-seed runs and the no-memory ablation never overwrite each other.
    """
    memory = '' if config['model']['task_transformer']['use_memory'] else '_no_memory'
    return f"{config['experiment']['name']}{memory}_seed{config['system']['seed']}"
