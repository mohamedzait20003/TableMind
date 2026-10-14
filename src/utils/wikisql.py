"""
WikiSQL dataset.

Loads the official WikiSQL release, executes each query on the official
SQLite database to get its gold answer, and yields TAPAS-tokenized
table-question inputs with T5 targets ("count: a | b" for aggregations).
"""

import os
import json
import torch
import tarfile
import dataclasses
import pandas as pd
import urllib.request
from torch.utils.data import Dataset
from typing import List, Dict, Any, Iterator, Optional, Tuple
from transformers import T5TokenizerFast, TapasTokenizer

from .sql import ANSWER_SEPARATOR, Aggregation, WikiSQLExecutor, format_target


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
    answer_values: List[str]   # gold denotation (official execution)
    target: str                # T5 target: cells, operator-prefixed if aggregated
    aggregation: Aggregation
    source_index: int          # line in the split's .jsonl file

    @property
    def answer(self) -> str:
        """Gold answer text: answer values joined by ANSWER_SEPARATOR."""
        return ANSWER_SEPARATOR.join(self.answer_values)


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


def wikisql_executor(split: str, cache_dir: Optional[str] = None) -> WikiSQLExecutor:
    """Executor on the split's official SQLite database."""
    name = _WIKISQL_FILES[split]
    return WikiSQLExecutor(os.path.join(_wikisql_dir(cache_dir), f'{name}.db'))


def load_wikisql(
    split: str, cache_dir: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """
    Yield WikiSQL examples as {'index', 'table_id', 'question', 'table', 'sql'}
    dicts, 'index' being the line number in the split's file.

    Reads the official GitHub release directly (no dataset script), so it
    works with any `datasets` version. Table cells are converted to strings.
    """
    name = _WIKISQL_FILES[split]
    data_dir = _wikisql_dir(cache_dir)
    with open(os.path.join(data_dir, f'{name}.tables.jsonl'), encoding='utf-8') as f:
        tables = {t['id']: t for t in map(json.loads, f)}
    with open(os.path.join(data_dir, f'{name}.jsonl'), encoding='utf-8') as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            table = tables[item['table_id']]
            yield {
                'index': index,
                'table_id': item['table_id'],
                'question': item['question'],
                'table': {
                    'header': table['header'],
                    'types': table['types'],
                    'rows': [[str(cell) for cell in row] for row in table['rows']],
                },
                'sql': item['sql'],
            }


def wikisql_memory_pairs(
    split: str = 'train', cache_dir: Optional[str] = None
) -> List[Tuple[int, str, str]]:
    """
    (index, question, query pattern) triples for the memory store.

    The pattern names the operation and columns ("count school / club team
    where player") but not the answer, so retrieving a similar question tells
    the model how to answer, not what the answer is.
    """
    pairs = []
    for item in load_wikisql(split, cache_dir):
        header, sql = item['table']['header'], item['sql']
        operation = Aggregation(sql['agg']).name.lower().replace('none', 'select')
        pattern = f"{operation} {header[sql['sel']]}"
        if sql['conds']:
            pattern += ' where ' + ', '.join(header[column] for column, _, _ in sql['conds'])
        pairs.append((item['index'], item['question'], pattern.lower()))
    return pairs


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
        answer_max_length: int = 64,
        subset_ratio: float = 1.0,
        cache_dir: Optional[str] = None,
        memory_ids: bool = False
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
            memory_ids: Return each example's memory-store id, so training
                never retrieves the example's own memory entry
        """
        raw_data = list(load_wikisql(split, cache_dir))
        self.executor = wikisql_executor(split, cache_dir)
        self.memory_ids = memory_ids
        self.raw_data = raw_data[:max(1, int(subset_ratio * len(raw_data)))]
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.answer_tokenizer = T5TokenizerFast.from_pretrained(answer_model_name)
        self.max_length = max_length
        self.answer_max_length = answer_max_length
        self.data = []

        self._preprocess_data()
        del self.executor  # sqlite connections can't go to DataLoader workers

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
            memory_ids=(stage == 'train'),
        )

    def _preprocess_data(self) -> None:
        """
        Execute each query on the official database. Examples whose gold
        answer is empty are kept (target: nothing, or "count:" etc.), so
        every test example is scored.
        """
        skipped, empty = 0, 0
        for item in self.raw_data:
            try:
                table, sql = item['table'], item['sql']
                aggregation = Aggregation(sql['agg'])
                answer_values = self.executor.execute(
                    item['table_id'], sql['sel'], sql['agg'], sql['conds']
                )
                cells = self.executor.select(item['table_id'], sql['sel'], sql['conds'])
                empty += not answer_values

                table_df = pd.DataFrame(
                    table['rows'],
                    columns=table['header']
                ).fillna("")

                self.data.append(WikiSQLExample(
                    question=item['question'],
                    table=table_df,
                    answer_values=answer_values,
                    target=format_target(aggregation, cells),
                    aggregation=aggregation,
                    source_index=item['index'],
                ))

            except (KeyError, IndexError, ValueError):
                skipped += 1

        print(f"WikiSQL: {len(self.data)} examples ({empty} with an empty gold "
              f"answer), skipped {skipped} malformed")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with tokenized table-question pair, T5 labels, the
            answer string and the answer values (for denotation accuracy)
        """
        example = self.data[idx]
        labels = tokenize_targets(
            self.answer_tokenizer, [example.target], self.answer_max_length
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
            'answer_values': example.answer_values,
            'memory_id': example.source_index if self.memory_ids else -1,
        }
