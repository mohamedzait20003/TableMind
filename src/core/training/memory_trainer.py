"""
Stage 1: memory training.

Trains the memory encoder on PAQ question-passage pairs with an in-batch
contrastive loss (each question's key must pick out its own passage's value),
then fills the key-value store: question keys -> passage values.
"""

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, Optional

from .base import BaseTrainer
from ..memory import MemoryEncoder, KeyValueMemoryStore


class MemoryTrainer(BaseTrainer):
    """Contrastive question -> passage training for the memory encoder."""

    def __init__(
        self,
        model: MemoryEncoder,
        train_dataloader: Optional[DataLoader],
        eval_dataloader: Optional[DataLoader] = None,
        temperature: float = 0.05,
        **kwargs
    ):
        """
        Initialize the memory trainer.

        Args:
            model: Memory encoder to train
            train_dataloader: PAQ training loader
            eval_dataloader: PAQ evaluation loader
            temperature: Softmax temperature for the contrastive logits
            **kwargs: Optimisation settings forwarded to BaseTrainer
        """
        super().__init__(model, train_dataloader, eval_dataloader, **kwargs)
        self.temperature = temperature

    def _logits(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Cosine similarity of every question key to every passage value."""
        question_keys, _ = self.model(
            batch['question_input_ids'].to(self.device),
            batch['question_attention_mask'].to(self.device),
        )
        _, passage_values = self.model(
            batch['passage_input_ids'].to(self.device),
            batch['passage_attention_mask'].to(self.device),
        )
        return (
            F.normalize(question_keys, dim=-1)
            @ F.normalize(passage_values, dim=-1).t()
            / self.temperature
        )

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """In-batch contrastive loss; the matching passage is the diagonal."""
        logits = self._logits(batch)
        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate contrastive loss and in-batch retrieval accuracy.

        Returns:
            {'eval_loss', 'eval_retrieval_accuracy'} or {} without eval data
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            logits = self._logits(batch)
            targets = torch.arange(logits.size(0), device=logits.device)
            total_loss += F.cross_entropy(logits, targets).item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()

        return {
            'eval_loss': total_loss / len(self.eval_dataloader),
            'eval_retrieval_accuracy': correct / total if total else 0.0,
        }

    @torch.no_grad()
    def populate_store(
        self, memory_store: KeyValueMemoryStore, dataloader: DataLoader
    ) -> None:
        """
        Encode PAQ pairs into the store: normalized question keys mapped to
        passage values.

        Args:
            memory_store: Store to fill (existing entries are kept)
            dataloader: PAQ loader (question and passage fields)
        """
        self.model.eval()
        for batch in tqdm(dataloader, desc="Encoding PAQ to Memory"):
            question_keys, _ = self.model(
                batch['question_input_ids'].to(self.device),
                batch['question_attention_mask'].to(self.device),
            )
            _, passage_values = self.model(
                batch['passage_input_ids'].to(self.device),
                batch['passage_attention_mask'].to(self.device),
            )
            question_keys = F.normalize(question_keys, dim=-1)
            for key, value in zip(question_keys, passage_values):
                memory_store.add_entry(key, value)

        print(f"Encoded {memory_store.size()} entries to memory store")
