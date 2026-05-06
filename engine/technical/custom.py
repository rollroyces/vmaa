#!/usr/bin/env python3
"""
VMAA Custom Indicator Builder
==============================
Build custom indicators from mathematical expressions.

Supports:
  - Arithmetic: +, -, *, /, (, )
  - Functions: MA, EMA, RSI, STD, MAX, MIN, ABS, LOG, POW
  - Column references: CLOSE, OPEN, HIGH, LOW, VOLUME
  - Persistence: save/load custom indicators as JSON
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.technical.config import TC
from engine.technical.indicators import sma, ema, rsi as _rsi, bollinger_bands

logger = logging.getLogger("vmaa.engine.technical.custom")

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

CUSTOM_STORAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_STORAGE_PATH = os.path.join(CUSTOM_STORAGE_DIR, "custom_indicators.json")

# Valid column references (case-insensitive)
VALID_COLUMNS = {"CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"}

# Valid functions and their signatures
VALID_FUNCTIONS = {
    "MA": {"args": 2, "desc": "MA(data, period) — Simple Moving Average"},
    "EMA": {"args": 2, "desc": "EMA(data, period) — Exponential Moving Average"},
    "RSI": {"args": 2, "desc": "RSI(data, period) — Relative Strength Index"},
    "STD": {"args": 2, "desc": "STD(data, period) — Standard Deviation"},
    "MAX": {"args": 2, "desc": "MAX(data, period) — Rolling Maximum"},
    "MIN": {"args": 2, "desc": "MIN(data, period) — Rolling Minimum"},
    "ABS": {"args": 1, "desc": "ABS(value) — Absolute value"},
    "LOG": {"args": 1, "desc": "LOG(value) — Natural logarithm"},
    "POW": {"args": 2, "desc": "POW(data, exponent) — Raise to power"},
    "ROLLING_SUM": {"args": 2, "desc": "ROLLING_SUM(data, period) — Rolling sum"},
    "CORR": {"args": 3, "desc": "CORR(data1, data2, period) — Rolling correlation"},
    "DIFF": {"args": 2, "desc": "DIFF(data, lag) — Difference over lag periods"},
}


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CustomIndicator:
    """A user-defined indicator built from a formula expression."""

    name: str                           # Unique name for this indicator
    formula: str                        # Mathematical expression
    description: str = ""               # Human-readable description
    created_at: str = ""                # ISO timestamp
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CustomIndicator":
        return cls(
            name=d["name"],
            formula=d["formula"],
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
            tags=d.get("tags", []),
        )


# ═══════════════════════════════════════════════════════════════════
# Formula Parser
# ═══════════════════════════════════════════════════════════════════

# Token types
_TOKEN_NUMBER = "NUMBER"
_TOKEN_COLUMN = "COLUMN"
_TOKEN_FUNC = "FUNC"
_TOKEN_OP = "OP"
_TOKEN_LPAREN = "LPAREN"
_TOKEN_RPAREN = "RPAREN"
_TOKEN_COMMA = "COMMA"


def _tokenize(formula: str) -> List[Tuple[str, str]]:
    """Tokenize a formula string into (type, value) pairs.

    Args:
        formula: Expression like "MA(CLOSE,20) - EMA(CLOSE,50)".

    Returns:
        List of (token_type, token_value) tuples.

    Raises:
        ValueError: If formula contains invalid tokens.
    """
    tokens = []
    i = 0
    n = len(formula)

    while i < n:
        ch = formula[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Numbers (including decimals)
        if ch.isdigit() or (ch == '.' and i + 1 < n and formula[i + 1].isdigit()):
            j = i
            while j < n and (formula[j].isdigit() or formula[j] == '.'):
                j += 1
            tokens.append((_TOKEN_NUMBER, formula[i:j]))
            i = j
            continue

        # Identifiers (column names, function names)
        if ch.isalpha():
            j = i
            while j < n and (formula[j].isalnum() or formula[j] == '_'):
                j += 1
            word = formula[i:j].upper()
            if word in VALID_FUNCTIONS:
                tokens.append((_TOKEN_FUNC, word))
            elif word in VALID_COLUMNS:
                tokens.append((_TOKEN_COLUMN, word))
            else:
                raise ValueError(f"Unknown identifier: '{formula[i:j]}'. Must be a column ({VALID_COLUMNS}) or function ({set(VALID_FUNCTIONS.keys())}).")
            i = j
            continue

        # Operators
        if ch in '+-*/':
            tokens.append((_TOKEN_OP, ch))
            i += 1
            continue

        # Parentheses
        if ch == '(':
            tokens.append((_TOKEN_LPAREN, '('))
            i += 1
            continue
        if ch == ')':
            tokens.append((_TOKEN_RPAREN, ')'))
            i += 1
            continue

        # Comma
        if ch == ',':
            tokens.append((_TOKEN_COMMA, ','))
            i += 1
            continue

        raise ValueError(f"Unexpected character at position {i}: '{ch}'")

    return tokens


def _validate_tokens(tokens: List[Tuple[str, str]]) -> None:
    """Validate token sequence for structural correctness.

    Raises ValueError with descriptive message if invalid.
    """
    if not tokens:
        raise ValueError("Formula is empty.")

    # Basic structure checks
    paren_count = 0
    for i, (tt, tv) in enumerate(tokens):
        if tt == _TOKEN_LPAREN:
            paren_count += 1
        elif tt == _TOKEN_RPAREN:
            paren_count -= 1
            if paren_count < 0:
                raise ValueError("Mismatched parentheses: unexpected ')'")

        # Check for consecutive operators
        if tt == _TOKEN_OP and i + 1 < len(tokens):
            next_tt = tokens[i + 1][0]
            if next_tt == _TOKEN_OP and tv != '-':
                raise ValueError(f"Consecutive operators at position {i}: '{tv}{tokens[i+1][1]}'")

        # Check no operator at start or end
        if tt == _TOKEN_OP and i == 0:
            raise ValueError(f"Formula cannot start with operator: '{tv}'")
        if tt == _TOKEN_OP and i == len(tokens) - 1:
            raise ValueError(f"Formula cannot end with operator: '{tv}'")

    if paren_count != 0:
        raise ValueError(f"Mismatched parentheses: {paren_count} unclosed '('")


def _validate_formula(formula: str) -> None:
    """Validate a formula string. Raises ValueError if invalid."""
    tokens = _tokenize(formula)
    _validate_tokens(tokens)


# ═══════════════════════════════════════════════════════════════════
# Formula Executor (Interpreter)
# ═══════════════════════════════════════════════════════════════════

class FormulaExecutor:
    """Executes a parsed formula against a DataFrame of OHLCV data.

    Uses a recursive descent approach — builds the result by walking
    the token tree and applying operations.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame.

        Args:
            df: DataFrame with Open/High/Low/Close/Volume columns.
        """
        self.df = df
        # Map column names to numpy arrays
        self._col_cache: Dict[str, np.ndarray] = {}
        self._get_column("CLOSE")  # pre-warm
        self._get_column("OPEN")
        self._get_column("HIGH")
        self._get_column("LOW")
        self._get_column("VOLUME")

    def _get_column(self, col: str) -> np.ndarray:
        """Get a price column as a numpy array."""
        col_upper = col.upper()
        if col_upper in self._col_cache:
            return self._col_cache[col_upper]

        # Look for the column in the DataFrame
        col_map = {
            "OPEN": ["Open", "open"],
            "HIGH": ["High", "high"],
            "LOW": ["Low", "low"],
            "CLOSE": ["Close", "close", "Adj Close"],
            "VOLUME": ["Volume", "volume"],
        }

        candidates = col_map.get(col_upper, [col_upper])
        found = None
        for c in candidates:
            if c in self.df.columns:
                found = c
                break

        if found is None:
            raise KeyError(f"Column '{col_upper}' not found in DataFrame. Available: {list(self.df.columns)}")

        arr = self.df[found].to_numpy(dtype=np.float64)
        self._col_cache[col_upper] = arr
        return arr

    def _call_function(self, func_name: str, args: List[np.ndarray]) -> np.ndarray:
        """Call a named function with the given arguments.

        Args:
            func_name: Function name (e.g., 'MA', 'RSI').
            args: List of numpy arrays as arguments.

        Returns:
            Result array.
        """
        n = len(self.df)

        if func_name == "MA":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 or len(args[1]) == 1 else args[1][-1]))
            return sma(data, period)

        elif func_name == "EMA":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            return ema(data, period)

        elif func_name == "RSI":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            return _rsi(data, period)

        elif func_name == "STD":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            result = np.full(len(data), np.nan, dtype=np.float64)
            p = period
            for i in range(p - 1, len(data)):
                result[i] = np.nanstd(data[i - p + 1 : i + 1])
            return result

        elif func_name == "MAX":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            result = np.full(len(data), np.nan, dtype=np.float64)
            p = period
            for i in range(p - 1, len(data)):
                result[i] = np.nanmax(data[i - p + 1 : i + 1])
            return result

        elif func_name == "MIN":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            result = np.full(len(data), np.nan, dtype=np.float64)
            p = period
            for i in range(p - 1, len(data)):
                result[i] = np.nanmin(data[i - p + 1 : i + 1])
            return result

        elif func_name == "ABS":
            data = args[0]
            return np.abs(data)

        elif func_name == "LOG":
            data = args[0]
            return np.log(np.where(data > 0, data, np.nan))

        elif func_name == "POW":
            data, exponent = args[0], float(args[1][0] if args[1].ndim == 0 else args[1][-1])
            return np.power(data, exponent)

        elif func_name == "ROLLING_SUM":
            data, period = args[0], int(round(float(args[1][0]) if args[1].ndim == 0 else args[1][-1]))
            result = np.full(len(data), np.nan, dtype=np.float64)
            p = period
            # Cumulative sum approach
            cs = np.cumsum(np.nan_to_num(data, 0))
            result[p - 1:] = cs[p - 1:] - np.concatenate(([0], cs[:-p]))
            return result

        elif func_name == "CORR":
            data1, data2, period = args[0], args[1], int(round(float(args[2][0] if args[2].ndim == 0 else args[2][-1])))
            result = np.full(len(data1), np.nan, dtype=np.float64)
            p = period
            for i in range(p - 1, len(data1)):
                w1 = data1[i - p + 1 : i + 1]
                w2 = data2[i - p + 1 : i + 1]
                mask = ~np.isnan(w1) & ~np.isnan(w2)
                if mask.sum() >= 2:
                    result[i] = np.corrcoef(w1[mask], w2[mask])[0, 1]
            return result

        elif func_name == "DIFF":
            data, lag = args[0], int(round(float(args[1][0] if args[1].ndim == 0 else args[1][-1])))
            result = np.full(len(data), np.nan, dtype=np.float64)
            result[lag:] = data[lag:] - data[:-lag]
            return result

        else:
            raise ValueError(f"Unknown function: {func_name}")

    def _parse_to_scalar(self, token_str: str) -> float:
        """Parse a NUMBER token to float."""
        return float(token_str)

    def _evaluate_expression(
        self, tokens: List[Tuple[str, str]], start: int
    ) -> Tuple[np.ndarray, int]:
        """Recursive descent evaluator.

        Grammar (simplified):
            expression → term (('+' | '-') term)*
            term       → factor (('*' | '/') factor)*
            factor     → ('+' | '-') factor | func_call | column | number | '(' expression ')'
            func_call  → FUNC '(' arg (',' arg)* ')'
            arg        → expression

        Returns:
            (result_array, next_token_index).
        """
        result, pos = self._parse_term(tokens, start)
        # expression → term (('+' | '-') term)*
        while pos < len(tokens) and tokens[pos][0] == _TOKEN_OP and tokens[pos][1] in '+-':
            op = tokens[pos][1]
            right, pos = self._parse_term(tokens, pos + 1)
            if op == '+':
                result = result + right
            else:
                result = result - right
        return result, pos

    def _parse_term(self, tokens: List[Tuple[str, str]], start: int) -> Tuple[np.ndarray, int]:
        """term → factor (('*' | '/') factor)*"""
        result, pos = self._parse_factor(tokens, start)
        while pos < len(tokens) and tokens[pos][0] == _TOKEN_OP and tokens[pos][1] in '*/':
            op = tokens[pos][1]
            right, pos = self._parse_factor(tokens, pos + 1)
            if op == '*':
                result = result * right
            else:
                # Division — avoid divide by zero
                result = np.divide(result, right, out=np.full_like(result, np.nan), where=(right != 0))
        return result, pos

    def _parse_factor(self, tokens: List[Tuple[str, str]], start: int) -> Tuple[np.ndarray, int]:
        """factor → unary_op factor | func_call | column | number | '(' expression ')'"""
        if start >= len(tokens):
            raise ValueError("Unexpected end of formula")

        tt, tv = tokens[start]
        n = len(self.df)

        # Unary minus
        if tt == _TOKEN_OP and tv == '-':
            result, pos = self._parse_factor(tokens, start + 1)
            return -result, pos

        # Unary plus (no-op)
        if tt == _TOKEN_OP and tv == '+':
            return self._parse_factor(tokens, start + 1)

        # Number
        if tt == _TOKEN_NUMBER:
            val = self._parse_to_scalar(tv)
            return np.full(n, val, dtype=np.float64), start + 1

        # Column reference
        if tt == _TOKEN_COLUMN:
            return self._get_column(tv).copy(), start + 1

        # Parenthesized expression
        if tt == _TOKEN_LPAREN:
            result, pos = self._evaluate_expression(tokens, start + 1)
            if pos >= len(tokens) or tokens[pos][0] != _TOKEN_RPAREN:
                raise ValueError("Expected ')' after expression")
            return result, pos + 1

        # Function call
        if tt == _TOKEN_FUNC:
            func_name = tv
            expected_args = VALID_FUNCTIONS[func_name]["args"]
            pos = start + 1  # skip FUNC token
            if pos >= len(tokens) or tokens[pos][0] != _TOKEN_LPAREN:
                raise ValueError(f"Expected '(' after function '{func_name}'")
            pos += 1  # skip '('

            # Parse arguments
            args = []
            first_arg = True
            while pos < len(tokens) and tokens[pos][0] != _TOKEN_RPAREN:
                if not first_arg:
                    if tokens[pos][0] != _TOKEN_COMMA:
                        raise ValueError(f"Expected ',' between function arguments, got {tokens[pos]}")
                    pos += 1
                first_arg = False
                arg_result, pos = self._evaluate_expression(tokens, pos)
                args.append(arg_result)

            if pos >= len(tokens) or tokens[pos][0] != _TOKEN_RPAREN:
                raise ValueError(f"Expected ')' after function arguments for '{func_name}'")
            pos += 1

            if len(args) != expected_args:
                raise ValueError(
                    f"Function '{func_name}' expects {expected_args} argument(s), got {len(args)}. "
                    f"Usage: {VALID_FUNCTIONS[func_name]['desc']}"
                )

            return self._call_function(func_name, args), pos

        raise ValueError(f"Unexpected token: {tokens[start]}")

    def evaluate(self, formula: str) -> np.ndarray:
        """Evaluate a formula against the DataFrame.

        Args:
            formula: Mathematical expression string.

        Returns:
            Numpy array with the computed indicator values.
        """
        tokens = _tokenize(formula)
        _validate_tokens(tokens)
        result, pos = self._evaluate_expression(tokens, 0)
        if pos != len(tokens):
            raise ValueError(f"Trailing tokens after expression at position {pos}")
        return result


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def validate_formula(formula: str) -> Tuple[bool, str]:
    """Validate a custom indicator formula.

    Args:
        formula: The formula string to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    try:
        _validate_formula(formula)
        return True, ""
    except ValueError as e:
        return False, str(e)


def custom_indicator(df: pd.DataFrame, formula: str) -> np.ndarray:
    """Compute a custom indicator from a formula string.

    Args:
        df: DataFrame with Open/High/Low/Close/Volume columns.
        formula: Mathematical expression. Examples:
            - "MA(CLOSE, 20) - MA(CLOSE, 50)"  → MA crossover diff
            - "(CLOSE - MA(CLOSE, 20)) / STD(CLOSE, 20) * 100"  → custom %B
            - "RSI(CLOSE, 14)"  → RSI
            - "MAX(HIGH, 20) - MIN(LOW, 20)"  → price range

    Returns:
        Numpy array with the computed indicator values (same length as df).
    """
    executor = FormulaExecutor(df)
    return executor.evaluate(formula)


def list_available_functions() -> Dict[str, str]:
    """Return a dict of all available functions and their descriptions."""
    return {name: info["desc"] for name, info in VALID_FUNCTIONS.items()}


# ═══════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════


def save_custom_indicators(indicators: List[CustomIndicator], path: Optional[str] = None) -> None:
    """Save custom indicators to a JSON file.

    Args:
        indicators: List of CustomIndicator objects.
        path: File path (defaults to CUSTOM_STORAGE_PATH).
    """
    filepath = path or CUSTOM_STORAGE_PATH
    data = [ind.to_dict() for ind in indicators]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(indicators)} custom indicators to {filepath}")


def load_custom_indicators(path: Optional[str] = None) -> List[CustomIndicator]:
    """Load custom indicators from a JSON file.

    Args:
        path: File path (defaults to CUSTOM_STORAGE_PATH).

    Returns:
        List of CustomIndicator objects.
    """
    filepath = path or CUSTOM_STORAGE_PATH
    if not os.path.exists(filepath):
        logger.debug(f"No custom indicators file at {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [CustomIndicator.from_dict(d) for d in data]


def add_custom_indicator(
    name: str, formula: str, description: str = "", tags: Optional[List[str]] = None
) -> CustomIndicator:
    """Create, validate, and persist a new custom indicator.

    Args:
        name: Unique name for the indicator.
        formula: Formula expression string.
        description: Human-readable description.
        tags: List of tags for search.

    Returns:
        The created CustomIndicator.

    Raises:
        ValueError: If formula is invalid or name already exists.
    """
    # Validate formula
    is_valid, error = validate_formula(formula)
    if not is_valid:
        raise ValueError(f"Invalid formula: {error}")

    # Check for duplicate name
    existing = load_custom_indicators()
    for ind in existing:
        if ind.name == name:
            raise ValueError(f"Custom indicator '{name}' already exists. Use a different name.")

    from datetime import datetime, timezone

    ind = CustomIndicator(
        name=name,
        formula=formula,
        description=description,
        created_at=datetime.now(timezone.utc).isoformat(),
        tags=tags or [],
    )

    existing.append(ind)

    # Enforce max count
    max_count = TC.max_custom_indicators
    if len(existing) > max_count:
        logger.warning(f"Custom indicator limit ({max_count}) exceeded. Oldest may be removed.")
        existing = existing[-max_count:]

    save_custom_indicators(existing)
    logger.info(f"Added custom indicator: {name}")
    return ind


def remove_custom_indicator(name: str) -> bool:
    """Remove a custom indicator by name.

    Args:
        name: Name of the indicator to remove.

    Returns:
        True if removed, False if not found.
    """
    existing = load_custom_indicators()
    original_len = len(existing)
    filtered = [ind for ind in existing if ind.name != name]
    if len(filtered) == original_len:
        return False
    save_custom_indicators(filtered)
    logger.info(f"Removed custom indicator: {name}")
    return True
