#!/usr/bin/env python3
"""
Main training script for DoT model.

This script provides a command-line interface for training DoT models
with configurable parameters and experiment tracking.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset
import random
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.memory import MemoryEncoder, KeyValueMemoryStore  # noqa: E402
from core.models import DoTModel  # noqa: E402
from data.datasets import PAQDataset, WikiSQLDataset  # noqa: E402
from training.trainer import DoTTrainer  # noqa: E402
from training.utils import collate_fn, encode_paq_to_memory  # noqa: E402


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict) -> tuple:
    """Create train and test datasets based on configuration."""
    # PAQ dataset for memory
    paq_dataset = PAQDataset(
        split=config['data']['paq']['split'],
        model_name=config['data']['paq']['model_name']
    )
    
    # Create PAQ subset
    paq_size = len(paq_dataset)
    paq_subset_size = max(1, int(config['data']['paq']['subset_ratio'] * paq_size))
    paq_indices = torch.randperm(paq_size).tolist()[:paq_subset_size]
    paq_subset = Subset(paq_dataset, paq_indices)
    
    # WikiSQL datasets
    train_dataset = WikiSQLDataset(
        split=config['data']['wikisql']['train_split'],
        model_name=config['data']['wikisql']['model_name'],
        max_length=config['data']['wikisql']['max_length']
    )
    
    test_dataset = WikiSQLDataset(
        split=config['data']['wikisql']['test_split'],
        model_name=config['data']['wikisql']['model_name'],
        max_length=config['data']['wikisql']['max_length']
    )
    
    # Create WikiSQL subsets
    train_size = len(train_dataset)
    train_subset_size = max(1, int(config['data']['wikisql']['train_subset_ratio'] * train_size))
    train_indices = list(range(train_subset_size))
    train_subset = Subset(train_dataset, train_indices)
    
    test_size = len(test_dataset)
    test_subset_size = max(1, int(config['data']['wikisql']['test_subset_ratio'] * test_size))
    test_indices = list(range(test_subset_size))
    test_subset = Subset(test_dataset, test_indices)
    
    return paq_subset, train_subset, test_subset


def create_model(config: dict, memory_store: KeyValueMemoryStore, memory_encoder: MemoryEncoder) -> DoTModel:
    """Create DoT model based on configuration."""
    model = DoTModel(
        memory_store=memory_store,
        memory_encoder=memory_encoder,
        k=config['model']['pruning_transformer']['top_k'],
        pruning_model=config['model']['pruning_transformer']['model_name'],
        task_model=config['model']['task_transformer']['model_name']
    )
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DoT model')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Override experiment name'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Override device selection'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.device:
        config['system']['device'] = args.device
    
    # Set device
    if config['system']['device'] == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config['system']['device']
    
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(config['system']['seed'])
    
    # Create datasets
    print("Creating datasets...")
    paq_subset, train_subset, test_subset = create_datasets(config)
    
    # Create data loaders
    paq_loader = DataLoader(
        paq_subset, 
        batch_size=config['training']['memory_batch_size'], 
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_subset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Dataset sizes - PAQ: {len(paq_subset)}, Train: {len(train_subset)}, Test: {len(test_subset)}")
    
    # Initialize memory components
    print("Initializing memory components...")
    memory_encoder = MemoryEncoder(
        model_name=config['model']['memory_encoder']['model_name'],
        hidden_size=config['model']['memory_encoder']['hidden_size'],
        proj_dim=config['model']['memory_encoder']['proj_dim']
    ).to(device)
    
    memory_store = KeyValueMemoryStore()
    
    # Encode PAQ to memory
    print("Encoding PAQ dataset to memory...")
    encode_paq_to_memory(memory_encoder, memory_store, paq_loader, device)
    
    # Create model
    print("Creating DoT model...")
    model = create_model(config, memory_store, memory_encoder)
    
    # Create trainer
    trainer = DoTTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        device=device,
        checkpoint_dir=config['system']['checkpoint_dir'],
        log_steps=config['training']['log_steps']
    )
    
    # Train model
    print("Starting training...")
    train_losses, eval_metrics = trainer.train(
        num_epochs=config['training']['num_epochs']
    )
    
    # Plot training curves
    print("Plotting training curves...")
    plot_path = os.path.join(
        config['system']['output_dir'], 
        f"{config['experiment']['name']}_training_curves.png"
    )
    os.makedirs(config['system']['output_dir'], exist_ok=True)
    trainer.plot_training_curves(save_path=plot_path)
    
    # Save final model
    final_checkpoint = os.path.join(
        config['system']['checkpoint_dir'],
        f"{config['experiment']['name']}_final.pt"
    )
    trainer.save_checkpoint(f"{config['experiment']['name']}_final")
    
    print(f"Training completed! Final checkpoint saved: {final_checkpoint}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    if eval_metrics:
        print(f"Final eval loss: {eval_metrics[-1]['eval_loss']:.4f}")
        print(f"Final eval accuracy: {eval_metrics[-1]['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()