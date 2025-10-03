"""
DoT Model Package - Differentiable Optimized Transformer

A modular implementation of the DoT architecture with memory-augmented transformers
for table question answering tasks.
"""

from .core.memory import MemoryEncoder, KeyValueMemoryStore
from .core.models import PruningTransformer, TaskTransformer, DoTModel
from .data.datasets import PAQDataset, WikiSQLDataset
from .training.trainer import DoTTrainer
from .training.utils import collate_fn, encode_paq_to_memory

__version__ = "0.1.0"
__author__ = "ECE-570 Project Team"

__all__ = [
    "MemoryEncoder",
    "KeyValueMemoryStore", 
    "PruningTransformer",
    "TaskTransformer",
    "DoTModel",
    "PAQDataset",
    "WikiSQLDataset",
    "DoTTrainer",
    "collate_fn",
    "encode_paq_to_memory",
]