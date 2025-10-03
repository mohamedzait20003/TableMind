"""
Training orchestration for DoT models.

This module provides a comprehensive trainer class that handles training,
evaluation, checkpointing, and logging for DoT models.
"""

import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from core.models import DoTModel


class DoTTrainer:
    """
    Comprehensive trainer for DoT models.
    
    Handles training loops, evaluation, checkpointing, and logging
    with configurable hyperparameters and callbacks.
    """
    
    def __init__(
        self,
        model: DoTModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_steps: int = 100,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: DoT model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            optimizer: Optimizer (creates AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory for saving checkpoints
            log_steps: Steps between logging
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_steps = log_steps
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=5e-5, 
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
            
        # Create scheduler if not provided
        if scheduler is None and train_dataloader is not None:
            total_steps = len(train_dataloader) * 3  # Assume 3 epochs default
            self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps,
            )
        else:
            self.scheduler = scheduler
            
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.train_losses = []
        self.eval_metrics = []

    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {self.epoch + 1}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_steps == 0:
                current_lr = (
                    self.scheduler.get_last_lr()[0] 
                    if self.scheduler else self.optimizer.param_groups[0]['lr']
                )
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Token-level accuracy
                preds = outputs.logits.argmax(dim=-1)
                correct_tokens += (preds == labels).sum().item()
                total_tokens += labels.numel()
        
        avg_loss = total_loss / len(self.eval_dataloader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
        }
        
        self.eval_metrics.append(metrics)
        return metrics

    def train(self, num_epochs: int = 3) -> Tuple[List[float], List[Dict]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Tuple of (train_losses, eval_metrics)
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")  # noqa: E501
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            # Logging
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"  # noqa: E501
            if eval_metrics:
                log_msg += f" - Eval Loss: {eval_metrics['eval_loss']:.4f}"
                log_msg += f" - Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}"  # noqa: E501
            print(log_msg)
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
        
        print("Training completed!")
        return self.train_losses, self.eval_metrics

    def save_checkpoint(self, name: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{name}.pt")
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.train_losses = checkpoint['train_losses']
        self.eval_metrics = checkpoint['eval_metrics']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:  # noqa: E501
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training curves.
        
        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        axes[0].plot(self.train_losses, 'b-', label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Evaluation metrics
        if self.eval_metrics:
            eval_losses = [m['eval_loss'] for m in self.eval_metrics]
            eval_accuracies = [m['eval_accuracy'] for m in self.eval_metrics]
            
            axes[1].plot(eval_losses, 'r-', label='Eval Loss')
            axes[1].set_ylabel('Loss', color='r')
            axes[1].tick_params(axis='y', labelcolor='r')
            
            ax2 = axes[1].twinx()
            ax2.plot(eval_accuracies, 'g-', label='Eval Accuracy')
            ax2.set_ylabel('Accuracy', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            axes[1].set_title('Evaluation Metrics')
            axes[1].set_xlabel('Epoch')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved: {save_path}")
        
        plt.show()