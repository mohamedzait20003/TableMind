"""
Answer-level evaluation for WikiSQL.

Denotation accuracy follows the weakly supervised WikiSQL protocol used by
TAPAS and DoT: a prediction is correct when its answer values equal the
values obtained by executing the gold SQL, as an unordered multiset, with
numbers compared numerically.
"""

import math
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from .sql import ANSWER_SEPARATOR, to_float

Value = Union[float, str]


def _normalize_value(value: str) -> Value:
    """Lower-case, drop whitespace, and parse numbers ("1,234" -> 1234.0)."""
    text = ''.join(value.lower().split())
    number = to_float(text)
    return number if number is not None and math.isfinite(number) else text


def _values_match(a: Value, b: Value) -> bool:
    """Numbers match within a relative tolerance, strings exactly."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    return a == b


def parse_prediction(text: str) -> List[str]:
    """Split a generated answer into values on ANSWER_SEPARATOR."""
    separator = ANSWER_SEPARATOR.strip()
    return [value for value in text.split(separator) if value.strip()]


def denotation_match(prediction: str, gold_values: List[str]) -> bool:
    """
    True when the predicted values equal the gold values as a multiset.

    Args:
        prediction: Generated answer text
        gold_values: Values from executing the gold SQL
    """
    predicted = [_normalize_value(v) for v in parse_prediction(prediction)]
    remaining = [_normalize_value(v) for v in gold_values]
    if len(predicted) != len(remaining):
        return False
    for value in predicted:
        match = next(
            (i for i, gold in enumerate(remaining) if _values_match(value, gold)),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    return True


def _exact(prediction: str, answer: str) -> bool:
    """String match ignoring case and whitespace."""
    return ''.join(prediction.lower().split()) == ''.join(answer.lower().split())


def evaluate_answers(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_new_tokens: int = 32,
    max_batches: Optional[int] = None
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """
    Generate answers and score them against the executed SQL answers.

    Args:
        model: DoT model with a `generate` method
        dataloader: WikiSQL loader built with `collate_fn`
        device: Device to run generation on
        max_new_tokens: Generation length limit
        max_batches: Stop after this many batches (None = all)

    Returns:
        ({'denotation_accuracy', 'exact_match', 'examples'}, per-example
        records with 'prediction', 'answer', 'denotation' and 'exact')
    """
    model.eval()
    records = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        predictions = model.generate(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
            max_new_tokens=max_new_tokens,
        )
        for prediction, answer, values in zip(
            predictions, batch['answers'], batch['answer_values']
        ):
            records.append({
                'prediction': prediction,
                'answer': answer,
                'denotation': denotation_match(prediction, values),
                'exact': _exact(prediction, answer),
            })

    total = len(records)
    metrics = {
        'denotation_accuracy': sum(r['denotation'] for r in records) / total if total else 0.0,
        'exact_match': sum(r['exact'] for r in records) / total if total else 0.0,
        'examples': total,
    }
    return metrics, records
