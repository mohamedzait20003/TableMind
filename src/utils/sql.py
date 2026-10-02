"""
WikiSQL query execution.

WikiSQL ships SQL queries, not answers; executing each query on its table
gives the answer values (the denotation) used as the T5 target.
"""

import enum
import dataclasses
from typing import Any, List, Optional

# Joins multi-row answers in T5 targets. A single T5 token that occurs in
# only ~0.03% of WikiSQL cells (", " collides with dates like "may 1, 1990").
ANSWER_SEPARATOR = " | "


class Aggregation(enum.Enum):
    """SQL aggregation operations."""
    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVERAGE = 5


class Operator(enum.Enum):
    """SQL comparison operators."""
    EQUALS = 0
    GREATER = 1
    LESSER = 2


@dataclasses.dataclass
class Condition:
    """SQL WHERE condition."""
    column: int
    operator: Operator
    cmp_value: Any


def to_float(value: Any) -> Optional[float]:
    """Parse a WikiSQL cell as a number ("1,234" -> 1234.0), or None."""
    try:
        return float(str(value).replace(',', '').strip())
    except ValueError:
        return None


def _format_number(value: float) -> str:
    """Render 3.0 as "3" and 2.5 as "2.5"."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip('0').rstrip('.')


def _condition_holds(cell: str, condition: Condition, is_real: bool) -> bool:
    """Evaluate one WHERE condition on a cell, case-insensitively."""
    cell_num, cmp_num = to_float(cell), to_float(condition.cmp_value)
    if condition.operator == Operator.EQUALS:
        if is_real and cell_num is not None and cmp_num is not None:
            return cell_num == cmp_num
        return str(cell).strip().lower() == str(condition.cmp_value).strip().lower()
    if cell_num is None or cmp_num is None:
        return False
    if condition.operator == Operator.GREATER:
        return cell_num > cmp_num
    return cell_num < cmp_num


def execute_wikisql(
    rows: List[List[str]],
    types: List[str],
    select_column: int,
    aggregation: Aggregation,
    conditions: List[Condition]
) -> Optional[List[str]]:
    """
    Execute a WikiSQL query on its table and return the answer values
    (the denotation).

    Args:
        rows: Table rows (lists of cell strings)
        types: Column types ('text' or 'real')
        select_column: Index of the SELECT column
        aggregation: Aggregation applied to the selected values
        conditions: WHERE conditions (ANDed)

    Returns:
        Lower-cased answer values (one per selected row, or a single
        aggregate), or None if the query has no answer
    """
    selected = [
        row[select_column] for row in rows
        if all(
            _condition_holds(row[c.column], c, types[c.column] == 'real')
            for c in conditions
        )
    ]

    if aggregation == Aggregation.COUNT:
        return [str(len(selected))]
    if aggregation == Aggregation.NONE:
        return [value.lower() for value in selected] if selected else None

    numbers = [n for n in (to_float(v) for v in selected) if n is not None]
    if not numbers:
        return None
    if aggregation == Aggregation.MAX:
        result = max(numbers)
    elif aggregation == Aggregation.MIN:
        result = min(numbers)
    elif aggregation == Aggregation.SUM:
        result = sum(numbers)
    else:
        result = sum(numbers) / len(numbers)
    return [_format_number(result)]
