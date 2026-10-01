"""
Stage 2: pretraining.

Trains the full DoT model (TAPAS pruner + memory-augmented T5) on WikiSQL,
with the memory store filled in stage 1.
"""

import torch
from tqdm import tqdm
from typing import Any, Dict

from .base import BaseTrainer


class PretrainTrainer(BaseTrainer):
    """Sequence-to-sequence training of the DoT model on WikiSQL answers."""

    def _forward(self, batch: Dict[str, Any]):
        """Run the DoT model on a collated WikiSQL batch."""
        return self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            token_type_ids=batch['token_type_ids'].to(self.device),
            labels=batch['labels'].to(self.device),
        )

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """T5 cross-entropy on the executed SQL answer."""
        return self._forward(batch).loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            {'eval_loss', 'eval_token_accuracy'} or {} without eval data;
            accuracy counts only real (non -100) label tokens
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss, correct_tokens, total_tokens = 0.0, 0, 0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            outputs = self._forward(batch)
            total_loss += outputs.loss.item()

            labels = batch['labels'].to(self.device)
            preds = outputs.logits.argmax(dim=-1)
            mask = labels != -100
            correct_tokens += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        return {
            'eval_loss': total_loss / len(self.eval_dataloader),
            'eval_token_accuracy': (
                correct_tokens / total_tokens if total_tokens else 0.0
            ),
        }
