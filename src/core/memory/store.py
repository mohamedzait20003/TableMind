"""
Key-value memory store for the DoT model.

Entries are kept on CPU; retrieval stacks them once and caches the stacked
tensors on the target device until the store changes.
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class KeyValueMemoryStore:
    """
    Simple key-value memory store for efficient retrieval.

    Entries are accumulated on CPU; on the first retrieval they are stacked
    once and cached on the target device until the store changes.
    """

    def __init__(self):
        """Initialize empty memory store."""
        self.keys = []
        self.values = []
        self._cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def add_entry(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """
        Add a key-value pair to the memory store.

        Args:
            key: Key tensor to store
            value: Value tensor to store
        """
        # Store on CPU to avoid GPU memory growth across runs
        self.keys.append(key.detach().cpu())
        self.values.append(value.detach().cpu())
        self._cache = None

    def _stacked(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) stacked on `device`, building the cache once."""
        device = torch.device(device)
        if self._cache is None or self._cache[0].device != device:
            self._cache = (
                torch.stack(self.keys).to(device),
                torch.stack(self.values).to(device),
            )
        return self._cache

    def retrieve(
        self,
        query_keys: torch.Tensor,
        top_k: int = 1,
        device: str = "cuda"
    ) -> Optional[torch.Tensor]:
        """
        Retrieve a softmax-weighted sum of the top-k most similar values.

        Gradients flow through the similarity scores into `query_keys`,
        so the query projection is trained (requires top_k > 1).

        Args:
            query_keys: Query keys of shape (batch_size, proj_dim)
            top_k: Number of top similar entries to retrieve
            device: Device to perform computations on

        Returns:
            Retrieved values of shape (batch_size, proj_dim) or None if empty
        """
        if len(self.keys) == 0:
            return None

        store_keys, store_values = self._stacked(device)
        top_k = min(top_k, store_keys.size(0))

        # Scaled dot-product similarity, as in attention
        sim = query_keys @ store_keys.t() / math.sqrt(store_keys.size(1))
        topk_sim, topk_idx = sim.topk(top_k, dim=1)
        weights = F.softmax(topk_sim, dim=1)

        # (batch, top_k, proj_dim) weighted by (batch, top_k, 1)
        return (weights.unsqueeze(-1) * store_values[topk_idx]).sum(dim=1)

    def size(self) -> int:
        """Return the number of stored entries."""
        return len(self.keys)

    def clear(self) -> None:
        """Clear all stored entries."""
        self.keys.clear()
        self.values.clear()
        self._cache = None

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return stored keys/values as stacked CPU tensors for checkpointing."""
        if len(self.keys) == 0:
            return {'keys': torch.empty(0), 'values': torch.empty(0)}
        return {
            'keys': torch.stack(self.keys),
            'values': torch.stack(self.values),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Replace the store contents with checkpointed keys/values."""
        self.clear()
        self.keys = list(state['keys'].cpu().unbind(0))
        self.values = list(state['values'].cpu().unbind(0))
