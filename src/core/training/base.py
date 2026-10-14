"""
Shared training loop for both DoT training stages.

Subclasses implement `compute_loss` and `evaluate`; checkpoint I/O lives in
`utils.checkpoint` and uses `state_dict` / `load_state_dict` from here.
Training can be checkpointed every few minutes and resumed mid-epoch.
"""

import math
import time
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from transformers import get_scheduler
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional, Tuple


def _get_rng_state() -> Dict[str, Any]:
    """Python, NumPy, CPU and CUDA RNG states (tensors/tuples only, so the
    checkpoint loads with torch.load(weights_only=True))."""
    np_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (np_state[0], torch.from_numpy(np_state[1].copy()), *np_state[2:]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states saved by `_get_rng_state`."""
    random.setstate(state['python'])
    name, keys, *rest = state['numpy']
    np.random.set_state((name, keys.cpu().numpy(), *rest))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state['cuda']])


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
        lr_multipliers: Optional[Dict[str, float]] = None,
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
            lr_multipliers: {name substring: factor}; matching parameters
                train at learning_rate * factor without weight decay
                (e.g. {'_gate': 10} for the memory gates)
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
            self._param_groups(float(learning_rate), lr_multipliers or {}),
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
        self.batches_done = 0       # batches finished in the current epoch
        self.epoch_loss_sum = 0.0   # summed loss of those batches
        self.train_losses: List[float] = []
        self.eval_metrics: List[Dict[str, float]] = []

        # Periodic checkpointing (configured by `train`)
        self._on_checkpoint: Optional[Callable[["BaseTrainer"], None]] = None
        self._checkpoint_seconds: Optional[float] = None
        self._last_checkpoint = time.monotonic()

    def _param_groups(
        self, learning_rate: float, lr_multipliers: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Default group plus one group per lr multiplier."""
        groups: Dict[str, List[nn.Parameter]] = {key: [] for key in lr_multipliers}
        default: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            key = next((k for k in lr_multipliers if k in name), None)
            (groups[key] if key is not None else default).append(param)
        param_groups: List[Dict[str, Any]] = [{'params': default}]
        for key, params in groups.items():
            if params:
                param_groups.append({
                    'params': params,
                    'lr': learning_rate * lr_multipliers[key],
                    'weight_decay': 0.0,
                })
        return param_groups

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

    def _checkpoint(self) -> None:
        """Call the checkpoint hook (if any) and restart the timer."""
        if self._on_checkpoint is not None:
            self._on_checkpoint(self)
        self._last_checkpoint = time.monotonic()

    def _checkpoint_due(self) -> bool:
        """True when the periodic checkpoint interval has elapsed."""
        return (
            self._checkpoint_seconds is not None
            and time.monotonic() - self._last_checkpoint >= self._checkpoint_seconds
        )

    def train_epoch(self) -> float:
        """
        Train for one epoch, continuing after `batches_done` when resuming.

        Mid-epoch resumption needs a sampler with `set_epoch(epoch, start)`
        (utils.ResumableSampler), which replays the epoch's order and skips
        the samples already trained on.

        Returns:
            Average training loss for the epoch
        """
        loader = self.train_dataloader
        start = self.batches_done
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(self.epoch, start * loader.batch_size)
        elif start:
            raise ValueError(
                "Resuming mid-epoch needs a ResumableSampler on the train loader"
            )

        self.model.train()
        self.optimizer.zero_grad()
        num_batches = start + len(loader)

        # Creating a DataLoader iterator draws a seed from the global RNG; a
        # resumed run creates an extra iterator mid-epoch, so keep the global
        # RNG untouched to continue exactly like an uninterrupted run
        rng_state = torch.get_rng_state()
        batches = iter(loader)
        torch.set_rng_state(rng_state)

        progress_bar = tqdm(
            batches,
            desc=f"Epoch {self.epoch + 1}",
            initial=start,
            total=num_batches,
        )

        for batch_idx, batch in enumerate(progress_bar, start=start + 1):
            loss = self.compute_loss(batch)
            (loss / self.accumulation_steps).backward()
            self.epoch_loss_sum += loss.item()

            if batch_idx % self.accumulation_steps == 0 or batch_idx == num_batches:
                self._optimizer_step()
                # Position only advances on optimizer steps, so a checkpoint
                # never holds half-accumulated gradients
                self.batches_done = batch_idx
                if self.global_step % self.log_steps == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'step': self.global_step
                    })
                if self._checkpoint_due():
                    self._checkpoint()

        self.batches_done = num_batches
        return self.epoch_loss_sum / max(1, num_batches)

    def train(
        self,
        num_epochs: Optional[int] = None,
        on_checkpoint: Optional[Callable[["BaseTrainer"], None]] = None,
        checkpoint_minutes: Optional[float] = None,
        on_epoch_end: Optional[Callable[["BaseTrainer"], None]] = None
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Train the model, continuing from the restored state if resumed.

        Args:
            num_epochs: Epochs to run in total (defaults to the constructor value)
            on_checkpoint: Called with the trainer every `checkpoint_minutes`
                (after an optimizer step) and after every epoch, e.g. to save
                and upload a resumable checkpoint
            checkpoint_minutes: Interval of mid-epoch checkpoints (None = only
                after each epoch)
            on_epoch_end: Called with the trainer after each epoch's
                evaluation, before the epoch checkpoint (e.g. to keep the
                best epoch)

        Returns:
            Tuple of (train_losses, eval_metrics)
        """
        num_epochs = num_epochs or self.num_epochs
        self._on_checkpoint = on_checkpoint
        self._checkpoint_seconds = checkpoint_minutes * 60 if checkpoint_minutes else None
        self._last_checkpoint = time.monotonic()

        if self.epoch >= num_epochs:
            print(f"Training already finished ({self.epoch}/{num_epochs} epochs)")
            return self.train_losses, self.eval_metrics
        if self.epoch or self.batches_done:
            print(f"Resuming at epoch {self.epoch + 1}, batch {self.batches_done} "
                  f"(step {self.global_step})")

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Training on {self.device} for {num_epochs} epoch(s), "
              f"{trainable:,} trainable parameters")

        while self.epoch < num_epochs:
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            eval_metrics = self.evaluate()
            if eval_metrics:
                self.eval_metrics.append(eval_metrics)

            log_msg = f"Epoch {self.epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
            for name, value in eval_metrics.items():
                log_msg += f" - {name}: {value:.4f}"
            print(log_msg)

            self.epoch += 1
            self.batches_done = 0
            self.epoch_loss_sum = 0.0
            if on_epoch_end is not None:
                on_epoch_end(self)
            self._checkpoint()

        return self.train_losses, self.eval_metrics

    def state_dict(self, include_optimizer: bool = True) -> Dict[str, Any]:
        """Training state: counters, position in the epoch, history, RNG
        states and optionally optimizer/scheduler state."""
        state = {
            'epoch': self.epoch,
            'batches_done': self.batches_done,
            'epoch_loss_sum': self.epoch_loss_sum,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'eval_metrics': self.eval_metrics,
            'rng': _get_rng_state(),
        }
        if include_optimizer:
            state['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                state['scheduler'] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore training state produced by `state_dict`."""
        self.epoch = state['epoch']
        self.batches_done = state.get('batches_done', 0)
        self.epoch_loss_sum = state.get('epoch_loss_sum', 0.0)
        self.global_step = state['global_step']
        self.train_losses = state['train_losses']
        self.eval_metrics = state['eval_metrics']
        if 'rng' in state:
            _set_rng_state(state['rng'])
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
