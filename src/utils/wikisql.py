"""
WikiSQL dataset.

Loads the official WikiSQL release, executes each query to get its answer,
and yields TAPAS-tokenized table-question inputs with T5 answer targets.
"""

import os
import json
import torch
import tarfile
import dataclasses
import pandas as pd
import urllib.request
from torch.utils.data import Dataset
from typing import List, Dict, Any, Iterator, Optional
from transformers import T5TokenizerFast, TapasTokenizer

from .sql import Aggregation, Operator, Condition, execute_wikisql


def tokenize_targets(
    tokenizer: T5TokenizerFast, texts: List[str], max_length: int
) -> torch.Tensor:
    """Tokenize target strings with T5, padding positions set to -100."""
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    labels = encoded['input_ids']
    labels[encoded['attention_mask'] == 0] = -100
    return labels


@dataclasses.dataclass
class WikiSQLExample:
    """Structured WikiSQL example."""
    question: str
    table: pd.DataFrame
    answer: str
    aggregation: Aggregation
    conditions: List[Condition]


_WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
_WIKISQL_FILES = {'train': 'train', 'validation': 'dev', 'dev': 'dev', 'test': 'test'}


def _wikisql_dir(cache_dir: Optional[str] = None) -> str:
    """Download and extract the official WikiSQL release once, return its data dir."""
    cache_dir = cache_dir or os.environ.get(
        'TABLEMIND_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'tablemind')
    )
    data_dir = os.path.join(cache_dir, 'data')
    if not os.path.exists(os.path.join(data_dir, 'test.tables.jsonl')):
        os.makedirs(cache_dir, exist_ok=True)
        archive = os.path.join(cache_dir, 'wikisql.tar.bz2')
        print(f"Downloading WikiSQL to {cache_dir} ...")
        urllib.request.urlretrieve(_WIKISQL_URL, archive)
        with tarfile.open(archive, 'r:bz2') as tar:
            tar.extractall(cache_dir)
        os.remove(archive)
    return data_dir


def load_wikisql(
    split: str, cache_dir: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """
    Yield WikiSQL examples as {'question', 'table', 'sql'} dicts.

    Reads the official GitHub release directly (no dataset script), so it
    works with any `datasets` version. Table cells are converted to strings.
    """
    name = _WIKISQL_FILES[split]
    data_dir = _wikisql_dir(cache_dir)
    with open(os.path.join(data_dir, f'{name}.tables.jsonl'), encoding='utf-8') as f:
        tables = {t['id']: t for t in map(json.loads, f)}
    with open(os.path.join(data_dir, f'{name}.jsonl'), encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            table = tables[item['table_id']]
            yield {
                'question': item['question'],
                'table': {
                    'header': table['header'],
                    'types': table['types'],
                    'rows': [[str(cell) for cell in row] for row in table['rows']],
                },
                'sql': item['sql'],
            }


class WikiSQLDataset(Dataset):
    """
    Dataset wrapper for WikiSQL dataset.

    Inputs are TAPAS-tokenized table-question pairs; targets are the answers
    obtained by executing each example's SQL, tokenized with T5.
    """

    def __init__(
        self,
        split: str = "train",
        model_name: str = 'google/tapas-large-finetuned-wtq',
        max_length: int = 512,
        answer_model_name: str = 't5-base',
        answer_max_length: int = 32,
        subset_ratio: float = 1.0,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize WikiSQL dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            model_name: Model name for TAPAS tokenizer
            max_length: Maximum TAPAS sequence length
            answer_model_name: Model name for the T5 target tokenizer
            answer_max_length: Maximum target length in T5 tokens
            subset_ratio: Fraction of the split to use (first examples)
            cache_dir: Where the WikiSQL release is downloaded
        """
        raw_data = list(load_wikisql(split, cache_dir))
        self.raw_data = raw_data[:max(1, int(subset_ratio * len(raw_data)))]
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.answer_tokenizer = T5TokenizerFast.from_pretrained(answer_model_name)
        self.max_length = max_length
        self.answer_max_length = answer_max_length
        self.data = []

        self._preprocess_data()

    @classmethod
    def from_config(cls, config: Dict[str, Any], stage: str) -> "WikiSQLDataset":
        """
        Build from a loaded config.

        Args:
            config: Loaded configuration
            stage: 'train', 'eval' or 'test' (selects split and subset ratio)
        """
        wikisql = config['data']['wikisql']
        return cls(
            split=wikisql[f'{stage}_split'],
            model_name=wikisql['model_name'],
            max_length=wikisql['max_length'],
            answer_model_name=config['model']['task_transformer']['model_name'],
            answer_max_length=wikisql['answer_max_length'],
            subset_ratio=wikisql[f'{stage}_subset_ratio'],
        )

    def _preprocess_data(self) -> None:
        """Preprocess raw WikiSQL data into structured examples."""
        skipped = 0
        for item in self.raw_data:
            try:
                table = item['table']
                sql = item['sql']
                aggregation = Aggregation(sql['agg'])
                conditions = [
                    Condition(
                        column=col_idx,
                        operator=Operator(op_idx),
                        cmp_value=cond_value,
                    )
                    for col_idx, op_idx, cond_value in sql['conds']
                ]

                answer = execute_wikisql(
                    table['rows'], table['types'], sql['sel'],
                    aggregation, conditions
                )
                if answer is None:
                    skipped += 1
                    continue

                table_df = pd.DataFrame(
                    table['rows'],
                    columns=table['header']
                ).fillna("")

                self.data.append(WikiSQLExample(
                    question=item['question'],
                    table=table_df,
                    answer=answer,
                    aggregation=aggregation,
                    conditions=conditions
                ))

            except (KeyError, IndexError, ValueError):
                skipped += 1

        print(f"WikiSQL: kept {len(self.data)} examples, skipped {skipped} "
              f"(malformed or empty answer)")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized table-question pair, T5 labels and
            the answer string
        """
        example = self.data[idx]
        labels = tokenize_targets(
            self.answer_tokenizer, [example.answer], self.answer_max_length
        ).squeeze(0)

        try:
            encoding = self.tokenizer(
                table=example.table,
                queries=example.question,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            # Fall back to the question alone for tables TAPAS cannot encode
            print(f"Error processing example {idx}: {e}")
            encoding = self.tokenizer(
                table=pd.DataFrame([[""]], columns=[""]),
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
            'labels': labels,
            'answer': example.answer,
        }
