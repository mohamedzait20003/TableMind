"""
Stage 2: pretraining.

Trains the full DoT model (TAPAS pruner + memory-augmented T5) on WikiSQL,
with the memory store filled in stage 1.
"""

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, Optional

from .base import BaseTrainer


class PretrainTrainer(BaseTrainer):
    """Sequence-to-sequence training of the DoT model on WikiSQL answers."""

    def __init__(
        self,
        model,
        train_dataloader: Optional[DataLoader],
        eval_dataloader: Optional[DataLoader] = None,
        denotation_fn: Optional[Callable[[str, List[str]], bool]] = None,
        max_new_tokens: int = 64,
        **kwargs
    ):
        """
        Args:
            model: DoT model
            train_dataloader: WikiSQL training loader
            eval_dataloader: WikiSQL validation loader
            denotation_fn: Scores a generated answer against the gold values
                (utils.denotation_match); when given, evaluation also reports
                'eval_denotation_accuracy' (used to pick the best epoch)
            max_new_tokens: Generation length limit for that evaluation
            **kwargs: Optimisation settings forwarded to BaseTrainer
        """
        super().__init__(model, train_dataloader, eval_dataloader, **kwargs)
        self.denotation_fn = denotation_fn
        self.max_new_tokens = max_new_tokens

    def _forward(self, batch: Dict[str, Any]):
        """Run the DoT model on a collated WikiSQL batch."""
        memory_ids = batch.get('memory_ids')
        return self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            token_type_ids=batch['token_type_ids'].to(self.device),
            labels=batch['labels'].to(self.device),
            memory_ids=memory_ids.to(self.device) if memory_ids is not None else None,
        )

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """T5 cross-entropy on the target (answer cells / operator)."""
        return self._forward(batch).loss

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            {'eval_loss', 'eval_token_accuracy'} plus
            'eval_denotation_accuracy' when `denotation_fn` is set, or {}
            without eval data; token accuracy counts only real (non -100)
            label tokens
        """
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss, correct_tokens, total_tokens = 0.0, 0, 0
        correct_answers, total_answers = 0, 0
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            outputs = self._forward(batch)
            total_loss += outputs.loss.item()

            labels = batch['labels'].to(self.device)
            preds = outputs.logits.argmax(dim=-1)
            mask = labels != -100
            correct_tokens += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            if self.denotation_fn is not None:
                predictions = self.model.generate(
                    batch['input_ids'].to(self.device),
                    batch['attention_mask'].to(self.device),
                    batch['token_type_ids'].to(self.device),
                    max_new_tokens=self.max_new_tokens,
                )
                correct_answers += sum(
                    self.denotation_fn(prediction, values)
                    for prediction, values in zip(predictions, batch['answer_values'])
                )
                total_answers += len(predictions)

        metrics = {
            'eval_loss': total_loss / len(self.eval_dataloader),
            'eval_token_accuracy': (
                correct_tokens / total_tokens if total_tokens else 0.0
            ),
        }
        if self.denotation_fn is not None:
            metrics['eval_denotation_accuracy'] = (
                correct_answers / total_answers if total_answers else 0.0
            )
        return metrics
