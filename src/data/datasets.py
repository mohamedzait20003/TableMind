"""
Dataset loading and preprocessing utilities.

This module provides dataset classes for PAQ and WikiSQL datasets
with proper tokenization and preprocessing for the DoT model.
"""

import enum
import dataclasses
from typing import List, Dict, Text, Any, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import T5Tokenizer, TapasTokenizer


class PAQDataset(Dataset):
    """
    Dataset wrapper for PAQ (Probably Asked Questions) dataset.
    
    Provides tokenized question-answer pairs for memory encoding
    and training purposes.
    """
    
    def __init__(self, split: str = 'train', model_name: str = 't5-base'):
        """
        Initialize PAQ dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            model_name: Model name for tokenizer initialization
        """
        self.dataset = load_dataset("embedding-data/PAQ_pairs", split=split)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with tokenized question and answer
        """
        item = self.dataset[idx]
        question = item.get('question', '')
        answer = item.get('answer', '')

        encoded_question = self.tokenizer(
            question,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        encoded_answer = self.tokenizer(
            answer,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": encoded_question['input_ids'].squeeze(0),
            "attention_mask": encoded_question['attention_mask'].squeeze(0),
            "labels": encoded_answer['input_ids'].squeeze(0),
        }


class _Aggregation(enum.Enum):
    """SQL aggregation operations."""
    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVERAGE = 5


class _Operator(enum.Enum):
    """SQL comparison operators."""
    EQUALS = 0
    GREATER = 1
    LESSER = 2


@dataclasses.dataclass
class _Condition:
    """SQL WHERE condition."""
    column: Text
    operator: _Operator
    cmp_value: Any


@dataclasses.dataclass
class WikiSQLExample:
    """Structured WikiSQL example."""
    question: str
    table: pd.DataFrame
    answer: str
    aggregation: _Aggregation
    conditions: List[_Condition]


class WikiSQLDataset(Dataset):
    """
    Dataset wrapper for WikiSQL dataset.
    
    Provides tokenized table-question pairs for training and evaluation
    of table question answering models.
    """
    
    def __init__(
        self, 
        split: str = "train",
        model_name: str = 'google/tapas-large-finetuned-wtq',
        max_length: int = 512
    ):
        """
        Initialize WikiSQL dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            model_name: Model name for TAPAS tokenizer
            max_length: Maximum sequence length
        """
        self.raw_data = load_dataset("wikisql", split=split)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.data = []
        
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """Preprocess raw WikiSQL data into structured examples."""
        for item in self.raw_data:
            try:
                question = item['question']
                table = item['table']
                answer = item['answer']['text'] if 'answer' in item else ""
                aggregation = _Aggregation(item['sql']['agg'])

                # Create DataFrame from table
                table_df = pd.DataFrame(
                    table['rows'], 
                    columns=table['header']
                )
                table_df = table_df.fillna("")

                # Parse conditions
                conditions = []
                if 'conds' in item['sql']:
                    conditions = [
                        _Condition(
                            column=table_df.columns[col_idx],
                            operator=_Operator(op_idx),
                            cmp_value=cond_value,
                        )
                        for col_idx, op_idx, cond_value in zip(
                            item['sql']['conds']['column_index'],
                            item['sql']['conds']['operator_index'],
                            item['sql']['conds']['condition'],
                        )
                    ]

                example = WikiSQLExample(
                    question=question,
                    table=table_df,
                    answer=answer,
                    aggregation=aggregation,
                    conditions=conditions
                )
                self.data.append(example)
                
            except (KeyError, IndexError, ValueError) as e:
                # Skip malformed examples
                print(f"Skipping malformed example: {e}")
                continue

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary with tokenized table-question pair
        """
        example = self.data[idx]
        
        try:
            encoding = self.tokenizer(
                table=example.table,
                queries=example.question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'token_type_ids': encoding['token_type_ids'].squeeze(0),
                'labels': encoding['input_ids'].squeeze(0),  # Self-supervised
            }
        except Exception as e:
            # Return dummy data for problematic examples
            print(f"Error processing example {idx}: {e}")
            dummy_encoding = self.tokenizer(
                table=pd.DataFrame([["dummy"]], columns=["col"]),
                queries="dummy question",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                'input_ids': dummy_encoding['input_ids'].squeeze(0),
                'attention_mask': dummy_encoding['attention_mask'].squeeze(0),
                'token_type_ids': dummy_encoding['token_type_ids'].squeeze(0),
                'labels': dummy_encoding['input_ids'].squeeze(0),
            }