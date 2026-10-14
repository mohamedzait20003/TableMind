"""
Answer-level evaluation for WikiSQL.

Denotation accuracy follows the weakly supervised WikiSQL protocol used by
TAPAS and DoT: a prediction is correct when its answer values equal the
values obtained by executing the gold SQL, as an unordered multiset, with
numbers compared numerically. Operator-prefixed predictions ("count: a | b")
are aggregated first. Examples with an empty gold answer are correct only
when the prediction is empty too.
"""

import math
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union

from .sql import ANSWER_SEPARATOR, answer_from_prediction, to_float

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


def denotation_match(prediction: str, gold_values: List[str]) -> bool:
    """
    True when the predicted values equal the gold values as a multiset.

    Args:
        prediction: Generated answer text
        gold_values: Values from executing the gold SQL
    """
    predicted = [_normalize_value(v) for v in answer_from_prediction(prediction)]
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
    """Computed answer text equals the gold answer text, ignoring case and
    whitespace (order-sensitive, unlike denotation accuracy)."""
    computed = ANSWER_SEPARATOR.join(answer_from_prediction(prediction))
    return ''.join(computed.lower().split()) == ''.join(answer.lower().split())


def evaluate_answers(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_new_tokens: int = 64,
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
        ({'denotation_accuracy', 'denotation_accuracy_answerable',
        'exact_match', 'examples', 'empty_gold'}, per-example records with
        'prediction' (generated text), 'computed' (answer after aggregation),
        'answer', 'empty_gold', 'denotation' and 'exact')
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
                'computed': ANSWER_SEPARATOR.join(answer_from_prediction(prediction)),
                'answer': answer,
                'empty_gold': not values,
                'denotation': denotation_match(prediction, values),
                'exact': _exact(prediction, answer),
            })

    def rate(rows: List[Dict[str, Any]], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    answerable = [r for r in records if not r['empty_gold']]
    metrics = {
        'denotation_accuracy': rate(records, 'denotation'),
        'denotation_accuracy_answerable': rate(answerable, 'denotation'),
        'exact_match': rate(records, 'exact'),
        'examples': len(records),
        'empty_gold': len(records) - len(answerable),
    }
    return metrics, records
