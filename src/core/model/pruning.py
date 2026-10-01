"""
Pruning transformer for the DoT model.

TAPAS scores every token of the flattened table-question input; the
highest-scoring tokens are kept for the task transformer.
"""

import torch
import torch.nn as nn
from transformers import TapasForQuestionAnswering, TapasTokenizer


class PruningTransformer(nn.Module):
    """
    Transformer model for pruning/scoring input candidates.

    Uses TAPAS model to score table-question pairs and select top-k
    candidates for further processing.
    """

    def __init__(self, model_name: str = 'google/tapas-large-finetuned-wtq'):
        """
        Initialize the pruning transformer.

        Args:
            model_name: HuggingFace model identifier for TAPAS model
        """
        super().__init__()
        self.tapas = TapasForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)

        self.register_buffer(
            'max_type_ids',
            torch.tensor(self.tapas.config.type_vocab_sizes) - 1,
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Score input candidates for pruning.

        Args:
            input_ids: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            token_type_ids: Token type ids for table structure

        Returns:
            Scores for each token position, shape (batch_size, seq_len)
        """
        # Use the cell-selection head directly. TAPAS's final logits divide
        # by a small temperature (~0.035) and push tokens outside the chosen
        # column to -10000, which saturates sigmoid(score) to exactly 0.
        hidden = self.tapas.tapas(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.minimum(token_type_ids, self.max_type_ids),
        ).last_hidden_state
        return hidden @ self.tapas.output_weights + self.tapas.output_bias
