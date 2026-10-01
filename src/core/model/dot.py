"""
Complete DoT (Double Transformer) model.

Joins the TAPAS pruner and the memory-augmented T5 task transformer:
selected TAPAS tokens are decoded to text, re-tokenized for T5, and their
pruning scores scale the T5 input embeddings.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput

from .task import TaskTransformer
from .pruning import PruningTransformer
from ..memory import MemoryEncoder, KeyValueMemoryStore


def _cell_marker(
    prev_cell: Optional[Tuple[int, int]], row: int, column: int
) -> Optional[str]:
    """Marker word to insert before a table token entering a new cell."""
    if prev_cell is None or row != prev_cell[0]:
        return 'header:' if row == 0 else 'row:'
    if column != prev_cell[1]:
        return ';'
    return None


class DoTModel(nn.Module):
    """
    Complete DoT (Double Transformer) model.

    The TAPAS pruner scores every token, the top-k (always including the
    question) are decoded back to text and re-tokenized for T5, and the
    pruning scores scale the T5 input embeddings so the pruner is trained
    end to end.
    """

    def __init__(
        self,
        memory_store: KeyValueMemoryStore,
        memory_encoder: MemoryEncoder,
        k: int = 128,
        pruning_model: str = 'google/tapas-large-finetuned-wtq',
        task_model: str = 't5-base',
        memory_top_k: int = 8,
        task_max_length: int = 512
    ):
        """
        Initialize the complete DoT model.

        Args:
            memory_store: Key-value memory store
            memory_encoder: Memory encoder
            k: Number of top TAPAS tokens to keep
            pruning_model: Model name for pruning transformer
            task_model: Model name for task transformer
            memory_top_k: Number of memory entries mixed per query
            task_max_length: Maximum T5 input length after re-tokenization
        """
        super().__init__()
        self.k = k
        self.task_max_length = task_max_length
        self.pruner = PruningTransformer(pruning_model)
        self.task_model = TaskTransformer(
            memory_store, memory_encoder, task_model, memory_top_k
        )
        self._skip_tokens = set(self.pruner.tokenizer.all_special_tokens)
        self._skip_tokens.add('[EMPTY]')

    def select_tokens(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Pick the top-k TAPAS positions, in their original order.

        Padding can never be selected and the question segment is always
        kept (segment id 0 in TAPAS token_type_ids[..., 0]).

        Returns:
            Sorted positions of shape (batch_size, min(k, seq_len))
        """
        is_real = attention_mask.bool()
        is_question = (token_type_ids[..., 0] == 0) & is_real
        selection = (
            scores.detach()
            .masked_fill(~is_real, float('-inf'))
            .masked_fill(is_question, float('inf'))
        )
        k = min(self.k, scores.size(1))
        topk_indices = selection.topk(k, dim=1).indices
        return topk_indices.sort(dim=1).values

    def _selected_words(
        self,
        tokens: List[str],
        positions: List[int],
        token_type_ids: List[List[int]],
        mask: List[int]
    ) -> Tuple[List[str], List[int]]:
        """
        Turn selected TAPAS word pieces back into words.

        Returns the word list (with structural markers) and, per word, the
        TAPAS position whose score scales it (-1 = question word or marker,
        left unscaled).
        """
        words, sources = ['question:'], [-1]
        prev_cell, prev_pos, last_is_token = None, None, False

        for token, pos, types, real in zip(tokens, positions, token_type_ids, mask):
            if not real or token in self._skip_tokens:
                continue
            segment, column, row = types[0], types[1], types[2]

            source = -1
            if segment == 1:
                marker = _cell_marker(prev_cell, row, column)
                if marker is not None:
                    words.append(marker)
                    sources.append(-1)
                    last_is_token = False
                prev_cell = (row, column)
                source = pos

            is_piece = token.startswith('##')
            text = token[2:] if is_piece else token
            if is_piece and last_is_token and prev_pos == pos - 1:
                words[-1] += text
            else:
                words.append(text)
                sources.append(source)
            prev_pos, last_is_token = pos, True

        return words, sources

    def retokenize(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        topk_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode the selected TAPAS tokens and re-tokenize them for T5.

        Returns:
            (t5_input_ids, t5_attention_mask, source_positions), where
            source_positions maps every T5 token to the TAPAS position whose
            pruning score scales it, or -1 for unscaled tokens.
        """
        selected_ids = input_ids.gather(1, topk_indices).tolist()
        selected_mask = attention_mask.gather(1, topk_indices).tolist()
        selected_types = token_type_ids.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, token_type_ids.size(-1))
        ).tolist()
        positions = topk_indices.tolist()

        batch_words, batch_sources = [], []
        for b in range(input_ids.size(0)):
            tokens = self.pruner.tokenizer.convert_ids_to_tokens(selected_ids[b])
            words, sources = self._selected_words(
                tokens, positions[b], selected_types[b], selected_mask[b]
            )
            batch_words.append(words)
            batch_sources.append(sources)

        encoding = self.task_model.tokenizer(
            batch_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.task_max_length,
            return_tensors='pt',
        )

        source_positions = torch.full_like(encoding['input_ids'], -1)
        for b, sources in enumerate(batch_sources):
            for t, word_idx in enumerate(encoding.word_ids(b)):
                if word_idx is not None:
                    source_positions[b, t] = sources[word_idx]

        device = input_ids.device
        return (
            encoding['input_ids'].to(device),
            encoding['attention_mask'].to(device),
            source_positions.to(device),
        )

    def task_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the pruning stage and build the task transformer inputs.

        Returns:
            (t5_input_ids, t5_attention_mask, token_scale); token_scale is
            sigmoid(pruning score) for table tokens and 1 elsewhere, and
            carries gradient back into the pruner.
        """
        scores = self.pruner(input_ids, attention_mask, token_type_ids)
        topk_indices = self.select_tokens(scores, attention_mask, token_type_ids)
        t5_ids, t5_mask, source_positions = self.retokenize(
            input_ids, attention_mask, token_type_ids, topk_indices
        )

        token_scores = scores.gather(1, source_positions.clamp(min=0))
        token_scale = torch.where(
            source_positions >= 0,
            torch.sigmoid(token_scores),
            torch.ones_like(token_scores),
        )
        return t5_ids, t5_mask, token_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the complete DoT pipeline.

        Args:
            input_ids: TAPAS token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: TAPAS token type ids (batch_size, seq_len, 7)
            labels: T5 target ids (-100 = ignored)

        Returns:
            Model outputs from task transformer
        """
        t5_ids, t5_mask, token_scale = self.task_inputs(
            input_ids, attention_mask, token_type_ids
        )
        return self.task_model(
            input_ids=t5_ids,
            attention_mask=t5_mask,
            token_scale=token_scale,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        max_new_tokens: int = 32,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate answer strings for a batch of table-question pairs.
        """
        t5_ids, t5_mask, token_scale = self.task_inputs(
            input_ids, attention_mask, token_type_ids
        )
        hidden = self.task_model.encode(t5_ids, t5_mask, token_scale)
        output_ids = self.task_model.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=t5_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        return self.task_model.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )
