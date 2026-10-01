"""
Shared training loop for both DoT training stages.

Subclasses implement `compute_loss` and `evaluate`; checkpoint I/O lives in
`utils.checkpoint` and uses `state_dict` / `load_state_dict` from here.
"""

import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple


class BaseTrainer:
    """
    Optimisation loop with AdamW, a warmup scheduler, gradient clipping,
    gradient accumulation and loss/metric history.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: Optional[DataLoader],
        eval_dataloader: Optional[DataLoader] = None,
        device: str = "cuda",
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "linear",
        gradient_clipping: float = 1.0,
        gradient_accumulation_steps: int = 1,
        log_steps: int = 100,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_dataloader: Training data loader (None for evaluation only)
            eval_dataloader: Evaluation data loader
            device: Device to train on
            num_epochs: Epochs to train; also sizes the LR schedule
            learning_rate: Peak learning rate
            weight_decay: AdamW weight decay
            warmup_ratio: Fraction of optimizer steps used for warmup
            scheduler_type: transformers scheduler name (linear, cosine, ...)
            gradient_clipping: Max gradient norm
            gradient_accumulation_steps: Batches per optimizer step
            log_steps: Optimizer steps between progress-bar updates
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clipping = gradient_clipping
        self.accumulation_steps = max(1, gradient_accumulation_steps)
        self.log_steps = log_steps

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

        self.scheduler = None
        if train_dataloader is not None:
            steps_per_epoch = math.ceil(len(train_dataloader) / self.accumulation_steps)
            total_steps = steps_per_epoch * num_epochs
            self.scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=int(warmup_ratio * total_steps),
                num_training_steps=total_steps,
            )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.train_losses: List[float] = []
        self.eval_metrics: List[Dict[str, float]] = []

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return the training loss for one batch."""
        raise NotImplementedError

    def evaluate(self) -> Dict[str, float]:
        """Return evaluation metrics on `eval_dataloader` ({} if none)."""
        raise NotImplementedError

    def _optimizer_step(self) -> None:
        """Clip, step optimizer and scheduler, reset gradients."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clipping
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}"
        )

        for batch_idx, batch in enumerate(progress_bar, start=1):
            loss = self.compute_loss(batch)
            (loss / self.accumulation_steps).backward()
            total_loss += loss.item()

            if batch_idx % self.accumulation_steps == 0 or batch_idx == num_batches:
                self._optimizer_step()
                if self.global_step % self.log_steps == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'step': self.global_step
                    })

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def train(
        self,
        num_epochs: Optional[int] = None,
        on_epoch_end: Optional[Callable[["BaseTrainer"], None]] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Train the model.

        Args:
            num_epochs: Epochs to run (defaults to the constructor value)
            on_epoch_end: Called with the trainer after each epoch, e.g. to
                save and upload a checkpoint

        Returns:
            Tuple of (train_losses, eval_metrics)
        """
        num_epochs = num_epochs or self.num_epochs
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training on {self.device} for {num_epochs} epoch(s), "
              f"{trainable:,} trainable parameters")

        for _ in range(num_epochs):
            train_loss = self.train_epoch()
            eval_metrics = self.evaluate()
            if eval_metrics:
                self.eval_metrics.append(eval_metrics)

            log_msg = f"Epoch {self.epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
            for name, value in eval_metrics.items():
                log_msg += f" - {name}: {value:.4f}"
            print(log_msg)

            self.epoch += 1
            if on_epoch_end is not None:
                on_epoch_end(self)

        return self.train_losses, self.eval_metrics

    def state_dict(self, include_optimizer: bool = True) -> Dict[str, Any]:
        """Training state (counters, history, optionally optimizer/scheduler)."""
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics,
        }
        if include_optimizer:
            state['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore training state produced by `state_dict`."""
        self.epoch = state['epoch']
        self.global_step = state['global_step']
        self.train_losses = state['train_losses']
        self.eval_metrics = state['eval_metrics']
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot training loss and evaluation metrics per epoch.

        Args:
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.train_losses, 'b-o', label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        if self.eval_metrics:
            for name in self.eval_metrics[0]:
                axes[1].plot(
                    [m[name] for m in self.eval_metrics], '-o', label=name
                )
            axes[1].set_title('Evaluation Metrics')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved: {save_path}")

        plt.show()
