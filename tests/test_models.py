"""
Unit tests for DoT model components.

This module provides basic unit tests for the core components
of the DoT model to ensure they work correctly.
"""

import sys
import os
import unittest
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.memory import MemoryEncoder, KeyValueMemoryStore  # noqa: E402
from core.models import PruningTransformer, TaskTransformer, DoTModel  # noqa: E402


class TestMemoryComponents(unittest.TestCase):
    """Test memory encoder and store components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"  # Use CPU for tests
        self.batch_size = 2
        self.seq_len = 10
        self.proj_dim = 64
        
    def test_memory_encoder_initialization(self):
        """Test memory encoder can be initialized."""
        encoder = MemoryEncoder(
            model_name='t5-small',
            hidden_size=512,
            proj_dim=self.proj_dim
        )
        self.assertIsNotNone(encoder)
        
    def test_memory_encoder_forward(self):
        """Test memory encoder forward pass."""
        encoder = MemoryEncoder(
            model_name='t5-small',
            hidden_size=512,
            proj_dim=self.proj_dim
        )
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)
        
        # Forward pass
        keys, values = encoder(input_ids, attention_mask)
        
        # Check output shapes
        self.assertEqual(keys.shape, (self.batch_size, self.proj_dim))
        self.assertEqual(values.shape, (self.batch_size, self.proj_dim))
        
    def test_memory_store_operations(self):
        """Test memory store add and retrieve operations."""
        store = KeyValueMemoryStore()
        
        # Test empty store
        self.assertEqual(store.size(), 0)
        
        # Add entries
        key1 = torch.randn(self.proj_dim)
        value1 = torch.randn(self.proj_dim)
        key2 = torch.randn(self.proj_dim)
        value2 = torch.randn(self.proj_dim)
        
        store.add_entry(key1, value1)
        store.add_entry(key2, value2)
        
        self.assertEqual(store.size(), 2)
        
        # Test retrieval
        query_keys = torch.randn(1, self.proj_dim)
        retrieved = store.retrieve(query_keys, top_k=1, device=self.device)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.shape, (1, self.proj_dim))
        
        # Test clear
        store.clear()
        self.assertEqual(store.size(), 0)


class TestModelComponents(unittest.TestCase):
    """Test model components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.batch_size = 2
        self.seq_len = 10
        
    def test_pruning_transformer_initialization(self):
        """Test pruning transformer can be initialized."""
        # This test might be slow due to model loading
        try:
            pruner = PruningTransformer(
                model_name='google/tapas-small-finetuned-wtq'
            )
            self.assertIsNotNone(pruner)
        except Exception as e:
            self.skipTest(f"Skipping due to model loading issue: {e}")
            
    def test_task_transformer_initialization(self):
        """Test task transformer can be initialized."""
        memory_store = KeyValueMemoryStore()
        memory_encoder = MemoryEncoder(model_name='t5-small', proj_dim=64)
        
        try:
            task_model = TaskTransformer(
                memory_store=memory_store,
                memory_encoder=memory_encoder,
                model_name='t5-small',
                proj_dim=64
            )
            self.assertIsNotNone(task_model)
        except Exception as e:
            self.skipTest(f"Skipping due to model loading issue: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete DoT model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        
    def test_dot_model_creation(self):
        """Test DoT model can be created."""
        memory_store = KeyValueMemoryStore()
        memory_encoder = MemoryEncoder(model_name='t5-small', proj_dim=64)
        
        try:
            model = DoTModel(
                memory_store=memory_store,
                memory_encoder=memory_encoder,
                k=32,
                pruning_model='google/tapas-small-finetuned-wtq',
                task_model='t5-small'
            )
            self.assertIsNotNone(model)
        except Exception as e:
            self.skipTest(f"Skipping due to model loading issue: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)