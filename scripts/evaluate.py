#!/usr/bin/env python3
"""
Simple evaluation script for trained DoT models.

This script loads a trained model checkpoint and evaluates it
on a test dataset.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.memory import MemoryEncoder, KeyValueMemoryStore  # noqa: E402
from core.models import DoTModel  # noqa: E402
from data.datasets import WikiSQLDataset  # noqa: E402
from training.trainer import DoTTrainer  # noqa: E402
from training.utils import collate_fn  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate DoT model')
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = WikiSQLDataset(
        split=config['data']['wikisql']['test_split'],
        model_name=config['data']['wikisql']['model_name'],
        max_length=config['data']['wikisql']['max_length']
    )
    
    # Create subset for evaluation
    test_size = len(test_dataset)
    test_subset_size = max(
        1, int(config['data']['wikisql']['test_subset_ratio'] * test_size)
    )
    test_indices = list(range(test_subset_size))
    test_subset = Subset(test_dataset, test_indices)
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize memory components (empty for evaluation)
    memory_encoder = MemoryEncoder(
        model_name=config['model']['memory_encoder']['model_name'],
        hidden_size=config['model']['memory_encoder']['hidden_size'],
        proj_dim=config['model']['memory_encoder']['proj_dim']
    ).to(device)
    
    memory_store = KeyValueMemoryStore()
    
    # Create model
    model = DoTModel(
        memory_store=memory_store,
        memory_encoder=memory_encoder,
        k=config['model']['pruning_transformer']['top_k'],
        pruning_model=config['model']['pruning_transformer']['model_name'],
        task_model=config['model']['task_transformer']['model_name']
    )
    
    # Create trainer for evaluation
    trainer = DoTTrainer(
        model=model,
        train_dataloader=None,
        eval_dataloader=test_loader,
        device=device
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint)
    
    # Evaluate
    print("Starting evaluation...")
    metrics = trainer.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Test Loss: {metrics['eval_loss']:.4f}")
    print(f"Test Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Test samples: {len(test_subset)}")


if __name__ == "__main__":
    main()