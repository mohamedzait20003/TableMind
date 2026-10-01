"""
PAQ (Probably Asked Questions) dataset.

Question-passage pairs used to train the memory encoder (stage 1) and to
fill the key-value memory store.
"""

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Any, Dict
from transformers import T5TokenizerFast


class PAQDataset(Dataset):
    """
    Dataset wrapper for PAQ (Probably Asked Questions) dataset.

    Streams a shuffled sample of `num_examples` question-passage pairs, so
    the full ~5 GB dataset is never downloaded.
    """

    def __init__(
        self,
        split: str = 'train',
        model_name: str = 't5-base',
        num_examples: int = 10000,
        max_length: int = 64,
        passage_max_length: int = 128,
        seed: int = 42
    ):
        """
        Initialize PAQ dataset.

        Args:
            split: Dataset split ('train' is the only split on the Hub)
            model_name: Model name for tokenizer initialization
            num_examples: Number of pairs to sample
            max_length: Fixed token length for questions
            passage_max_length: Fixed token length for answer passages
            seed: Shuffle seed for the streamed sample
        """
        stream = load_dataset(
            "embedding-data/PAQ_pairs", split=split, streaming=True
        )
        stream = stream.shuffle(seed=seed, buffer_size=10_000)
        # Each row is {"set": [question, answer_passage]}
        self.pairs = [
            (row['set'][0], row['set'][1]) for row in stream.take(num_examples)
        ]
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.max_length = max_length
        self.passage_max_length = passage_max_length

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PAQDataset":
        """Build from a loaded config (tokenizer = memory encoder's)."""
        paq = config['data']['paq']
        return cls(
            split=paq['split'],
            model_name=config['model']['memory_encoder']['model_name'],
            num_examples=paq['num_examples'],
            max_length=paq['max_length'],
            passage_max_length=paq['passage_max_length'],
            seed=config['system']['seed'],
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.pairs)

    def _encode(self, text: str, max_length: int) -> Dict[str, torch.Tensor]:
        """Tokenize one string to a fixed length."""
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_length,
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized question and passage
        """
        question, passage = self.pairs[idx]
        encoded_question = self._encode(question, self.max_length)
        encoded_passage = self._encode(passage, self.passage_max_length)

        return {
            "question_input_ids": encoded_question['input_ids'],
            "question_attention_mask": encoded_question['attention_mask'],
            "passage_input_ids": encoded_passage['input_ids'],
            "passage_attention_mask": encoded_passage['attention_mask'],
        }
