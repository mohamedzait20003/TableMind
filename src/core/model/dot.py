"""
Complete DoT (Double Transformer) model.

Joins the TAPAS pruner and the memory-augmented T5 task transformer:
whole table cells are selected by their pruning scores, decoded to text,
re-tokenized for T5, and their scores scale the T5 input embeddings.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutput

from .task import TaskTransformer
from .pruning import PruningTransformer
from ..memory import MemoryEncoder, KeyValueMemoryStore

# TAPAS row/column ids are < 256 (their embedding sizes)
_MAX_IDS = 256


def _cell_marker(
    prev_cell: Optional[Tuple[int, int]], row: int, column: int
) -> Optional[str]:
    """Marker word to insert before a table token entering a new cell."""
    if prev_cell is None or row != prev_cell[0]:
        return 'header:' if row == 0 else 'row:'
    if column != prev_cell[1]:
        return ';'
    return None


def _add_piece(
    words: List[str], sources: List[int], token: str, source: int, follows: bool
) -> None:
    """
    Append a TAPAS word piece; a '##' piece that directly follows the
    previous token in the same cell is merged into the previous word.
    """
    if token.startswith('##') and follows:
        words[-1] += token[2:]
    else:
        words.append(token.removeprefix('##'))
        sources.append(source)


def _fill_budget(cell_index: List[int], scores: List[float], budget: int) -> List[int]:
    """
    Greedily pick whole cells, highest score first, within a token budget.

    Args:
        cell_index: Cell id per position (0 = not a selectable cell)
        scores: Cell score per position
        budget: Number of tokens available

    Returns:
        Positions of the selected cells' tokens
    """
    cells: Dict[int, List[int]] = {}
    for pos, cell in enumerate(cell_index):
        if cell:
            cells.setdefault(cell, []).append(pos)

    selected = []
    for positions in sorted(cells.values(), key=lambda p: scores[p[0]], reverse=True):
        if budget <= 0:
            break
        if len(positions) <= budget:
            selected.extend(positions)
            budget -= len(positions)
    return selected


class DoTModel(nn.Module):
    """
    Complete DoT (Double Transformer) model.

    The TAPAS pruner scores every token; whole cells are kept in order of
    their mean score until k TAPAS tokens are used (the question and the
    header row are always kept). Kept tokens are decoded back to text and
    re-tokenized for T5, and sigmoid(cell score) scales the T5 input
    embeddings so the pruner is trained end to end.
    """

    def __init__(
        self,
        memory_store: KeyValueMemoryStore,
        memory_encoder: MemoryEncoder,
        k: int = 256,
        pruning_model: str = 'google/tapas-small-finetuned-wtq',
        task_model: str = 't5-base',
        memory_top_k: int = 8,
        task_max_length: int = 512,
        use_memory: bool = True
    ):
        """
        Initialize the complete DoT model.

        Args:
            memory_store: Key-value memory store
            memory_encoder: Memory encoder
            k: TAPAS-token budget kept for the task transformer
            pruning_model: Model name for pruning transformer
            task_model: Model name for task transformer
            memory_top_k: Number of memory entries mixed per query
            task_max_length: Maximum T5 input length after re-tokenization
            use_memory: Retrieve from the memory store (False = ablation)
        """
        super().__init__()
        self.k = k
        self.task_max_length = task_max_length
        self.pruner = PruningTransformer(pruning_model)
        self.task_model = TaskTransformer(
            memory_store, memory_encoder, task_model, memory_top_k, use_memory
        )
        self._skip_tokens = set(self.pruner.tokenizer.all_special_tokens)
        self._skip_tokens.add('[EMPTY]')

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        memory_store: KeyValueMemoryStore,
        memory_encoder: MemoryEncoder
    ) -> "DoTModel":
        """Build from a loaded config (model and pretraining sections)."""
        model = config['model']
        return cls(
            memory_store,
            memory_encoder,
            k=model['pruning_transformer']['top_k'],
            pruning_model=model['pruning_transformer']['model_name'],
            task_model=model['task_transformer']['model_name'],
            memory_top_k=config['pretraining']['memory_top_k'],
            task_max_length=model['task_transformer']['max_length'],
            use_memory=model['task_transformer']['use_memory'],
        )

    @staticmethod
    def _cell_index(token_type_ids: torch.Tensor, is_cell: torch.Tensor) -> torch.Tensor:
        """Unique id per (row, column) cell; 0 for tokens outside cells."""
        row = token_type_ids[..., 2].clamp(max=_MAX_IDS - 1)
        column = token_type_ids[..., 1].clamp(max=_MAX_IDS - 1)
        return torch.where(is_cell, row * _MAX_IDS + column + 1, torch.zeros_like(row))

    def cell_scores(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Replace each table token's score by the mean score of its cell.

        Differentiable (scatter_add), so every token in a cell receives
        gradient. Question tokens keep their own scores.

        Returns:
            Scores of shape (batch_size, seq_len)
        """
        is_cell = (token_type_ids[..., 0] == 1) & attention_mask.bool()
        cell_index = self._cell_index(token_type_ids, is_cell)
        num_cells = _MAX_IDS * _MAX_IDS + 1

        sums = scores.new_zeros(scores.size(0), num_cells).scatter_add(1, cell_index, scores)
        counts = scores.new_zeros(scores.size(0), num_cells).scatter_add(
            1, cell_index, is_cell.to(scores.dtype)
        )
        means = sums / counts.clamp(min=1.0)
        return torch.where(is_cell, means.gather(1, cell_index), scores)

    def select_tokens(
        self,
        cell_scores: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Choose the TAPAS positions passed to the task transformer.

        The question and header row are always kept; body cells are added
        whole, highest cell score first, while they fit in the k-token
        budget. Padding is never selected.

        Returns:
            Boolean keep mask of shape (batch_size, seq_len)
        """
        is_real = attention_mask.bool()
        segment, row = token_type_ids[..., 0], token_type_ids[..., 2]
        is_body = is_real & (segment == 1) & (row > 0)
        keep = is_real & ~is_body  # question + header row

        cell_index = self._cell_index(token_type_ids, is_body).tolist()
        scores = cell_scores.detach().tolist()
        for b in range(keep.size(0)):
            budget = self.k - int(keep[b].sum())
            keep[b, _fill_budget(cell_index[b], scores[b], budget)] = True
        return keep

    def _selected_words(
        self,
        tokens: List[str],
        positions: List[int],
        token_type_ids: List[List[int]]
    ) -> Tuple[List[str], List[int]]:
        """
        Turn selected TAPAS word pieces back into words.

        Returns the word list (with structural markers) and, per word, the
        TAPAS position whose cell score scales it (-1 = question, header or
        marker, left unscaled).
        """
        words, sources = ['question:'], [-1]
        prev_cell, prev_pos = None, None

        for token, pos, types in zip(tokens, positions, token_type_ids):
            if token in self._skip_tokens:
                continue
            segment, column, row = types[0], types[1], types[2]

            marker = None
            if segment == 1:
                marker = _cell_marker(prev_cell, row, column)
                prev_cell = (row, column)
            if marker is not None:
                words.append(marker)
                sources.append(-1)

            # Only body cells are scaled by their pruning score
            source = pos if segment == 1 and row > 0 else -1
            follows = marker is None and prev_pos == pos - 1
            _add_piece(words, sources, token, source, follows)
            prev_pos = pos

        return words, sources

    def retokenize(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        keep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode the kept TAPAS tokens and re-tokenize them for T5.

        Returns:
            (t5_input_ids, t5_attention_mask, source_positions), where
            source_positions maps every T5 token to the TAPAS position whose
            cell score scales it, or -1 for unscaled tokens.
        """
        batch_words, batch_sources = [], []
        for b in range(input_ids.size(0)):
            positions = keep[b].nonzero().squeeze(-1)
            tokens = self.pruner.tokenizer.convert_ids_to_tokens(
                input_ids[b, positions].tolist()
            )
            words, sources = self._selected_words(
                tokens, positions.tolist(), token_type_ids[b, positions].tolist()
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
            sigmoid(cell score) for body-cell tokens and 1 elsewhere, and
            carries gradient back into the pruner.
        """
        scores = self.pruner(input_ids, attention_mask, token_type_ids)
        cell_scores = self.cell_scores(scores, attention_mask, token_type_ids)
        keep = self.select_tokens(cell_scores, attention_mask, token_type_ids)
        t5_ids, t5_mask, source_positions = self.retokenize(
            input_ids, token_type_ids, keep
        )

        token_scores = cell_scores.gather(1, source_positions.clamp(min=0))
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
