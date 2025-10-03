#!/usr/bin/env python3
"""
Quick demonstration script for the reorganized DoT model.

This script demonstrates the key features of the new architecture:
- Configuration-driven setup
- Modular component initialization
- Professional training pipeline
- Comprehensive logging and checkpointing
"""

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader, Subset

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.memory import MemoryEncoder, KeyValueMemoryStore  # noqa: E402
from core.models import DoTModel  # noqa: E402
from data.datasets import PAQDataset, WikiSQLDataset  # noqa: E402
from training.trainer import DoTTrainer  # noqa: E402
from training.utils import collate_fn, encode_paq_to_memory  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def demo_modular_setup():
    """Demonstrate the modular architecture setup."""
    print("🔧 Demonstrating Modular Architecture Setup")
    print("=" * 50)
    
    # Load configuration
    config = load_config('config/quick.yaml')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    print(f"⚙️  Configuration: {config['experiment']['name']}")
    
    # Initialize memory components
    print("\n🧠 Initializing Memory Components...")
    memory_encoder = MemoryEncoder(
        model_name=config['model']['memory_encoder']['model_name'],
        proj_dim=config['model']['memory_encoder']['proj_dim']
    ).to(device)
    
    memory_store = KeyValueMemoryStore()
    print(f"   ✅ Memory encoder: {config['model']['memory_encoder']['model_name']}")
    print(f"   ✅ Memory store initialized (size: {memory_store.size()})")
    
    # Create datasets with error handling
    print("\n📊 Creating Datasets...")
    try:
        # PAQ dataset for memory (very small subset for demo)
        paq_dataset = PAQDataset(
            split='train',
            model_name=config['model']['memory_encoder']['model_name']
        )
        
        # Use tiny subset for demo
        paq_size = min(10, len(paq_dataset))  # Maximum 10 examples
        paq_subset = Subset(paq_dataset, list(range(paq_size)))
        paq_loader = DataLoader(paq_subset, batch_size=2, shuffle=False)
        
        print(f"   ✅ PAQ dataset: {len(paq_subset)} examples")
        
        # Encode to memory
        print("   🔄 Encoding PAQ to memory...")
        encode_paq_to_memory(memory_encoder, memory_store, paq_loader, device)
        print(f"   ✅ Memory populated: {memory_store.size()} entries")
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        print("   ℹ️  This is expected if datasets aren't available")
        return
    
    # Create WikiSQL dataset
    try:
        wikisql_dataset = WikiSQLDataset(
            split='train',
            model_name=config['data']['wikisql']['model_name']
        )
        
        # Use tiny subset for demo
        wiki_size = min(5, len(wikisql_dataset))  # Maximum 5 examples
        wiki_subset = Subset(wikisql_dataset, list(range(wiki_size)))
        wiki_loader = DataLoader(wiki_subset, batch_size=1, collate_fn=collate_fn)
        
        print(f"   ✅ WikiSQL dataset: {len(wiki_subset)} examples")
        
    except Exception as e:
        print(f"   ❌ WikiSQL loading failed: {e}")
        print("   ℹ️  This is expected if datasets aren't available")
        return
    
    # Create DoT model
    print("\n🤖 Creating DoT Model...")
    try:
        model = DoTModel(
            memory_store=memory_store,
            memory_encoder=memory_encoder,
            k=config['model']['pruning_transformer']['top_k'],
            pruning_model=config['model']['pruning_transformer']['model_name'],
            task_model=config['model']['task_transformer']['model_name']
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✅ DoT model created: {param_count:,} parameters")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        print("   ℹ️  This is expected if models aren't available")
        return
    
    # Create trainer
    print("\n🏋️  Creating Professional Trainer...")
    trainer = DoTTrainer(
        model=model,
        train_dataloader=wiki_loader,
        eval_dataloader=wiki_loader,
        device=device,
        checkpoint_dir='./demo_checkpoints',
        log_steps=1
    )
    
    # Demonstrate trainer capabilities
    param_count = sum(p.numel() for p in trainer.model.parameters())
    print(f"   ✅ Trainer ready with {param_count:,} parameters")
    print(f"   📁 Checkpoints: ./demo_checkpoints")
    
    print("\n🎯 Architecture Demo Complete!")
    print("   ✨ Modular components ready for experimentation")
    print("   ⚙️  Configuration-driven setup working")
    print("   🔬 Professional training infrastructure initialized")
    

def demo_configuration_system():
    """Demonstrate the configuration system."""
    print("\n⚙️  Configuration System")
    print("=" * 25)
    
    config_file = 'config/default.yaml'
    if os.path.exists(config_file):
        config = load_config(config_file)
        print(f"✅ Config loaded: {config['experiment']['name']}")
        print(f"📊 Ready for {config['training']['num_epochs']} epochs")
    else:
        print("❌ Configuration file not found")


def main():
    """Run the demonstration."""
    print("🚀 DoT Model Architecture Demo")
    print("=" * 35)
    print("Showcasing the reorganized project structure\n")
    
    # Demo modular setup
    demo_modular_setup()
    
    # Demo configuration system
    demo_configuration_system()
    
    print("\n🎉 Demo Complete!")
    print("Next: python scripts/train.py --config config/quick.yaml")


if __name__ == "__main__":
    main()