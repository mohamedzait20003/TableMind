"""
WikiSQL query execution and aggregation targets.

Gold answers come from executing each query on WikiSQL's official SQLite
databases with the semantics of the official evaluator (lib/dbengine.py).
For aggregation queries the T5 target is the operator plus the selected
cells ("count: a | b"); the aggregate is then computed in Python.
"""

import re
import enum
import sqlite3
import dataclasses
from typing import Any, Dict, List, Optional, Sequence

# Joins multi-row answers in T5 targets. A single T5 token that occurs in
# only ~0.03% of WikiSQL cells (", " collides with dates like "may 1, 1990").
ANSWER_SEPARATOR = " | "

_AGG_SQL = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
_COND_SQL = ['=', '>', '<', 'OP']
_NUM_RE = re.compile(r'[-+]?\d*\.\d+|\d+')
_SCHEMA_RE = re.compile(r'\((.+)\)')
_SQLITE_NUMBER_RE = re.compile(r'^\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')


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


# Operator words used in T5 targets ("count: a | b")
_TARGET_WORDS = {
    Aggregation.MAX: 'max',
    Aggregation.MIN: 'min',
    Aggregation.COUNT: 'count',
    Aggregation.SUM: 'sum',
    Aggregation.AVERAGE: 'average',
}
_WORD_TO_AGG = {word: agg for agg, word in _TARGET_WORDS.items()}
_PREFIX_RE = re.compile(r'^\s*(' + '|'.join(_WORD_TO_AGG) + r')\s*:(.*)$', re.S)


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


def _format_value(value: Any) -> str:
    """Render a database value as answer text."""
    if isinstance(value, float):
        return _format_number(value)
    return str(value).strip()


def _plain_float(value: str) -> Optional[float]:
    """Parse a plain number (no thousands separators), or None."""
    try:
        return float(value)
    except ValueError:
        return None


def _sqlite_number(value: str) -> float:
    """SQLite's numeric conversion for SUM/AVG: longest numeric prefix, else 0."""
    match = _SQLITE_NUMBER_RE.match(value)
    return float(match.group(0)) if match else 0.0


class WikiSQLExecutor:
    """
    Executes WikiSQL queries on an official SQLite database (train.db,
    dev.db, test.db), porting DBEngine.execute from WikiSQL's lib/dbengine.py.
    """

    def __init__(self, db_path: str):
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._schemas: Dict[str, Dict[str, str]] = {}

    def _schema(self, table: str) -> Dict[str, str]:
        if table not in self._schemas:
            sql = self.connection.execute(
                'SELECT sql FROM sqlite_master WHERE tbl_name = ?', (table,)
            ).fetchone()[0]
            self._schemas[table] = dict(
                part.split(' ') for part in _SCHEMA_RE.findall(sql)[0].split(', ')
            )
        return self._schemas[table]

    def _run(
        self, table_id: str, select_sql: str, conditions: Sequence[Sequence[Any]]
    ) -> List[str]:
        table = 'table_{}'.format(table_id.replace('-', '_'))
        schema = self._schema(table)
        where, params = [], {}
        for column, operator, value in conditions:
            if isinstance(value, str):
                value = value.lower()
            if schema['col{}'.format(column)] == 'real' and not isinstance(value, (int, float)):
                try:
                    value = float(str(value).replace(',', ''))
                except ValueError:
                    value = float(_NUM_RE.findall(value)[0])
            where.append('col{} {} :col{}'.format(column, _COND_SQL[operator], column))
            params['col{}'.format(column)] = value
        query = 'SELECT {} AS result FROM {}'.format(select_sql, table)
        if where:
            query += ' WHERE ' + ' AND '.join(where)
        rows = self.connection.execute(query, params).fetchall()
        return [_format_value(row[0]) for row in rows if row[0] is not None]

    def execute(
        self, table_id: str, select_column: int, aggregation: int,
        conditions: Sequence[Sequence[Any]]
    ) -> List[str]:
        """Gold answer values (the denotation); empty if the query has none."""
        select = 'col{}'.format(select_column)
        if _AGG_SQL[aggregation]:
            select = '{}({})'.format(_AGG_SQL[aggregation], select)
        return self._run(table_id, select, conditions)

    def select(
        self, table_id: str, select_column: int, conditions: Sequence[Sequence[Any]]
    ) -> List[str]:
        """Values of the selected column before aggregation."""
        return self._run(table_id, 'col{}'.format(select_column), conditions)


def apply_aggregation(aggregation: Aggregation, values: List[str]) -> List[str]:
    """
    Aggregate cell values like SQLite does: COUNT counts values, MAX/MIN
    compare numerically when all values are numbers (text sorts after
    numbers), SUM/AVG read each value's numeric prefix.
    """
    if aggregation == Aggregation.NONE:
        return values
    if aggregation == Aggregation.COUNT:
        return [str(len(values))]
    if not values:
        return []
    if aggregation in (Aggregation.MAX, Aggregation.MIN):
        # Plain numbers only: "1,234" comes from a text column, which SQLite
        # compares as text
        numbers = [_plain_float(v) for v in values]
        texts = [v for v, n in zip(values, numbers) if n is None]
        numeric = [n for n in numbers if n is not None]
        if aggregation == Aggregation.MAX:
            return [max(texts)] if texts else [_format_number(max(numeric))]
        return [_format_number(min(numeric))] if numeric else [min(texts)]
    total = sum(_sqlite_number(v) for v in values)
    if aggregation == Aggregation.SUM:
        return [_format_number(total)]
    return [_format_number(total / len(values))]


def format_target(aggregation: Aggregation, selected_values: List[str]) -> str:
    """T5 target: the cells, prefixed by the operator for aggregations."""
    cells = ANSWER_SEPARATOR.join(selected_values)
    if aggregation == Aggregation.NONE:
        return cells
    return f"{_TARGET_WORDS[aggregation]}: {cells}".rstrip()


def split_values(text: str) -> List[str]:
    """Split answer text into values on ANSWER_SEPARATOR."""
    separator = ANSWER_SEPARATOR.strip()
    return [value.strip() for value in text.split(separator) if value.strip()]


def answer_from_prediction(text: str) -> List[str]:
    """
    Answer values for a generated target: "count: a | b" -> ["2"],
    "a | b" -> ["a", "b"].
    """
    match = _PREFIX_RE.match(text)
    if match is None:
        return split_values(text)
    return apply_aggregation(_WORD_TO_AGG[match.group(1)], split_values(match.group(2)))
