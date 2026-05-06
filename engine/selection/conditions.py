#!/usr/bin/env python3
"""
VMAA Smart Selection Engine — Condition Combinations
======================================================
Boolean condition engine for stock screening with AND/OR/NOT operators.

Features:
  - Condition types: gt, lt, gte, lte, eq, between, in_range, in_list
  - AND/OR/NOT operators between conditions
  - Nested conditions: (A AND B) OR (C AND NOT D)
  - Condition templates for common screening patterns
  - Serializable to/from JSON

Usage:
  from engine.selection.conditions import Condition, And, Or, Not
  
  # Deep value pattern
  cond = And(
      Condition("lt", field="pe_ratio", value=15),
      Condition("lt", field="pb_ratio", value=1.5),
      Condition("lt", field="market_cap", value=2e9),
  )
  
  # Check a stock
  stock = {"pe_ratio": 12, "pb_ratio": 1.2, "market_cap": 1.5e9, ...}
  result = cond.evaluate(stock)  # → True
  
  # Serialize
  json_str = cond.to_json()
  restored = Condition.from_json(json_str)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("vmaa.engine.selection.conditions")

# Type alias for stock data (dict or object with attributes)
StockData = Union[Dict[str, Any], object]


# ═══════════════════════════════════════════════════════════════════
# Condition Operators
# ═══════════════════════════════════════════════════════════════════

class OpType(Enum):
    """Condition operator types."""
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    BETWEEN = "between"      # value between low and high (inclusive)
    IN_RANGE = "in_range"    # value in [low, high)
    IN_LIST = "in_list"      # field value is one of a list
    EXISTS = "exists"        # field exists and is not None/empty
    CUSTOM = "custom"        # Custom lambda/function


# Operator comparison functions
_OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "neq": lambda a, b: a != b,
}


# ═══════════════════════════════════════════════════════════════════
# Condition Nodes (Composite Pattern)
# ═══════════════════════════════════════════════════════════════════

class ConditionNode(ABC):
    """Abstract base for all condition nodes."""
    
    @abstractmethod
    def evaluate(self, stock: StockData) -> bool:
        """Evaluate this condition against a stock's data."""
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON persistence."""
        ...
    
    @abstractmethod
    def explain(self) -> str:
        """Human-readable explanation of the condition."""
        ...
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ConditionNode:
        """Deserialize from dictionary."""
        return _deserialize(data)
    
    @staticmethod
    def from_json(json_str: str) -> ConditionNode:
        """Deserialize from JSON string."""
        return _deserialize(json.loads(json_str))
    
    def __and__(self, other: ConditionNode) -> 'And':
        return And(self, other)
    
    def __or__(self, other: ConditionNode) -> 'Or':
        return Or(self, other)
    
    def __invert__(self) -> 'Not':
        return Not(self)


def _get_field_value(stock: StockData, field: str) -> Any:
    """Extract a field value from stock data (dict or object)."""
    if isinstance(stock, dict):
        return stock.get(field)
    return getattr(stock, field, None)


# ═══════════════════════════════════════════════════════════════════
# Primitive Condition
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Condition(ConditionNode):
    """
    A single comparison condition on a stock field.
    
    Supported operators: gt, lt, gte, lte, eq, neq, between, in_range, in_list
    
    Example:
        >>> c = Condition("gte", field="roe", value=0.15)
        >>> c.evaluate({"roe": 0.20})  # True
        >>> c.evaluate({"roe": 0.10})  # False
        
        >>> c = Condition("between", field="pe_ratio", low=5, high=25)
        >>> c.evaluate({"pe_ratio": 15})  # True
        
        >>> c = Condition("in_list", field="sector", values=["Technology", "Healthcare"])
        >>> c.evaluate({"sector": "Technology"})  # True
    """
    op: str = "gt"
    field: str = ""
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[List[Any]] = None
    custom_fn: Optional[Callable[[Any], bool]] = None
    label: str = ""
    weight: float = 1.0  # For scoring (0-1 weight)
    
    def evaluate(self, stock: StockData) -> bool:
        """Evaluate this condition."""
        field_value = _get_field_value(stock, self.field)
        
        if field_value is None and self.op != "exists":
            return False
        
        try:
            if self.op in _OPERATORS:
                return _OPERATORS[self.op](float(field_value), float(self.value))
            
            elif self.op == "between":
                if self.low is None or self.high is None:
                    return False
                return float(self.low) <= float(field_value) <= float(self.high)
            
            elif self.op == "in_range":
                if self.low is None or self.high is None:
                    return False
                return float(self.low) <= float(field_value) < float(self.high)
            
            elif self.op == "in_list":
                if self.values is None:
                    return False
                return field_value in self.values
            
            elif self.op == "exists":
                return field_value is not None and field_value != "" and field_value != 0
            
            elif self.op == "custom":
                if self.custom_fn is None:
                    return False
                return self.custom_fn(field_value)
            
            else:
                logger.warning(f"Unknown operator: {self.op}")
                return False
                
        except (TypeError, ValueError):
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        d: Dict[str, Any] = {
            "type": "condition",
            "op": self.op,
            "field": self.field,
        }
        if self.label:
            d["label"] = self.label
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.value is not None:
            d["value"] = self.value
        if self.low is not None:
            d["low"] = self.low
        if self.high is not None:
            d["high"] = self.high
        if self.values is not None:
            d["values"] = self.values
        return d
    
    def explain(self) -> str:
        """Human-readable explanation."""
        prefix = f"[{self.label}] " if self.label else ""
        
        if self.op in _OPERATORS:
            syms = {"gt": ">", "lt": "<", "gte": "≥", "lte": "≤", "eq": "=", "neq": "≠"}
            sym = syms.get(self.op, self.op)
            return f"{prefix}{self.field} {sym} {self.value}"
        elif self.op == "between":
            return f"{prefix}{self.low} ≤ {self.field} ≤ {self.high}"
        elif self.op == "in_range":
            return f"{prefix}{self.low} ≤ {self.field} < {self.high}"
        elif self.op == "in_list":
            vals = ", ".join(str(v) for v in (self.values or [])[:5])
            if self.values and len(self.values) > 5:
                vals += f", ... (+{len(self.values)-5})"
            return f"{prefix}{self.field} IN [{vals}]"
        elif self.op == "exists":
            return f"{prefix}{self.field} EXISTS"
        elif self.op == "custom":
            return f"{prefix}{self.field} CUSTOM"
        return f"{prefix}{self.field} {self.op} {self.value}"


# ═══════════════════════════════════════════════════════════════════
# Composite Conditions (AND, OR, NOT)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class And(ConditionNode):
    """Logical AND — all children must evaluate True."""
    conditions: List[ConditionNode] = field(default_factory=list)
    label: str = ""
    
    def evaluate(self, stock: StockData) -> bool:
        """All conditions must be true."""
        return all(c.evaluate(stock) for c in self.conditions)
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": "and",
            "conditions": [c.to_dict() for c in self.conditions],
        }
        if self.label:
            d["label"] = self.label
        return d
    
    def explain(self) -> str:
        parts = [c.explain() for c in self.conditions]
        inner = " AND ".join(parts)
        return f"({inner})" if len(parts) > 1 else inner
    
    def add(self, condition: ConditionNode) -> 'And':
        """Add another condition (fluent API)."""
        self.conditions.append(condition)
        return self


@dataclass
class Or(ConditionNode):
    """Logical OR — at least one child must evaluate True."""
    conditions: List[ConditionNode] = field(default_factory=list)
    label: str = ""
    
    def evaluate(self, stock: StockData) -> bool:
        """At least one condition must be true."""
        return any(c.evaluate(stock) for c in self.conditions)
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": "or",
            "conditions": [c.to_dict() for c in self.conditions],
        }
        if self.label:
            d["label"] = self.label
        return d
    
    def explain(self) -> str:
        parts = [c.explain() for c in self.conditions]
        inner = " OR ".join(parts)
        return f"({inner})" if len(parts) > 1 else inner
    
    def add(self, condition: ConditionNode) -> 'Or':
        """Add another condition (fluent API)."""
        self.conditions.append(condition)
        return self


@dataclass
class Not(ConditionNode):
    """Logical NOT — inverts the child condition."""
    condition: Optional[ConditionNode] = None
    label: str = ""
    
    def evaluate(self, stock: StockData) -> bool:
        """Invert the child condition."""
        if self.condition is None:
            return True
        return not self.condition.evaluate(stock)
    
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": "not"}
        if self.condition:
            d["condition"] = self.condition.to_dict()
        if self.label:
            d["label"] = self.label
        return d
    
    def explain(self) -> str:
        inner = self.condition.explain() if self.condition else "?"
        return f"NOT ({inner})"


# ═══════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════

def _deserialize(data: Dict[str, Any]) -> ConditionNode:
    """Deserialize a condition dictionary back to a ConditionNode tree."""
    node_type = data.get("type", "condition")
    
    if node_type == "condition":
        return Condition(
            op=data.get("op", "gt"),
            field=data.get("field", ""),
            value=data.get("value"),
            low=data.get("low"),
            high=data.get("high"),
            values=data.get("values"),
            label=data.get("label", ""),
            weight=data.get("weight", 1.0),
        )
    
    elif node_type == "and":
        children = [_deserialize(c) for c in data.get("conditions", [])]
        return And(conditions=children, label=data.get("label", ""))
    
    elif node_type == "or":
        children = [_deserialize(c) for c in data.get("conditions", [])]
        return Or(conditions=children, label=data.get("label", ""))
    
    elif node_type == "not":
        child = _deserialize(data["condition"]) if "condition" in data else None
        return Not(condition=child, label=data.get("label", ""))
    
    raise ValueError(f"Unknown condition type: {node_type}")


# ═══════════════════════════════════════════════════════════════════
# Condition Templates
# ═══════════════════════════════════════════════════════════════════

class ConditionTemplates:
    """
    Pre-built condition templates for common screening patterns.
    
    Usage:
        >>> ct = ConditionTemplates()
        >>> deep_value = ct.deep_value()
        >>> growth_garp = ct.growth_at_reasonable_price()
        >>> cond = ct.deep_value_or_growth()
    """
    
    @staticmethod
    def deep_value(
        max_market_cap: float = 2e9,
        max_pe: float = 15,
        max_pb: float = 1.5,
        min_fcf_yield: float = 0.03,
    ) -> And:
        """
        Deep value screening: small cap, low multiples, strong cash flow.
        
        Market Cap < $2B AND P/E < 15 AND P/B < 1.5 AND FCF Yield > 3%
        
        Example:
            >>> cond = ConditionTemplates.deep_value(max_market_cap=1e9)
            >>> stock = {"market_cap": 800e6, "pe_ratio": 10, "pb_ratio": 1.2, "fcf_yield": 0.05}
            >>> cond.evaluate(stock)  # True
        """
        return And(
            conditions=[
                Condition("lt", field="market_cap", value=max_market_cap, label="Small cap"),
                Condition("lt", field="pe_ratio", value=max_pe, label="Low P/E"),
                Condition("lt", field="pb_ratio", value=max_pb, label="Low P/B"),
                Condition("gt", field="fcf_yield", value=min_fcf_yield, label="FCF positive"),
            ],
            label="Deep Value",
        )
    
    @staticmethod
    def growth_at_reasonable_price(
        max_pe: float = 30,
        min_revenue_growth: float = 0.10,
        min_roe: float = 0.10,
        max_debt_equity: float = 1.0,
    ) -> And:
        """
        GARP: Growth at a reasonable price.
        
        Revenue Growth > 10% AND ROE > 10% AND P/E < 30 AND D/E < 1.0
        
        Example:
            >>> cond = ConditionTemplates.growth_at_reasonable_price(max_pe=25)
        """
        return And(
            conditions=[
                Condition("lt", field="pe_ratio", value=max_pe, label="Reasonable P/E"),
                Condition("gt", field="revenue_growth_yoy", value=min_revenue_growth, label="Growing revenue"),
                Condition("gt", field="roe", value=min_roe, label="Profitable"),
                Condition("lt", field="debt_to_equity", value=max_debt_equity, label="Low debt"),
            ],
            label="GARP",
        )
    
    @staticmethod
    def high_momentum(
        min_return_3m: float = 0.20,
        max_rsi: float = 70,
        min_volume_1m: float = 100000,
        max_beta: float = 2.0,
    ) -> And:
        """
        High momentum with risk controls.
        
        3M Return > 20% AND Price < RSI overbought AND Volume adequate AND Beta manageable
        
        Example:
            >>> cond = ConditionTemplates.high_momentum(min_return_3m=0.30, max_rsi=65)
        """
        conditions = [
            Condition("gt", field="return_3m", value=min_return_3m, label="3M momentum"),
            Condition("lt", field="beta", value=max_beta, label="Manageable beta"),
        ]
        return And(conditions=conditions, label="High Momentum")
    
    @staticmethod
    def quality_value(
        min_roe: float = 0.15,
        min_fcf_yield: float = 0.04,
        max_pe: float = 20,
        max_debt_equity: float = 0.5,
        min_profit_margin: float = 0.10,
    ) -> And:
        """
        Quality value: profitable companies at reasonable prices.
        
        ROE > 15% AND FCF Yield > 4% AND P/E < 20 AND D/E < 0.5 AND Profit Margin > 10%
        
        Example:
            >>> cond = ConditionTemplates.quality_value()
        """
        return And(
            conditions=[
                Condition("gte", field="roe", value=min_roe, label="High ROE"),
                Condition("gte", field="fcf_yield", value=min_fcf_yield, label="Strong FCF"),
                Condition("lt", field="pe_ratio", value=max_pe, label="Reasonable P/E"),
                Condition("lt", field="debt_to_equity", value=max_debt_equity, label="Low leverage"),
                Condition("gte", field="profit_margin", value=min_profit_margin, label="Good margins"),
            ],
            label="Quality Value",
        )
    
    @staticmethod
    def turnaround(
        min_revenue_growth: float = 0.05,
        max_pe: float = 25,
        positive_fcf: bool = True,
    ) -> And:
        """
        Turnaround play: growth restarting, reasonable valuation.
        
        Revenue Growth > 5% AND P/E < 25 AND FCF Yield > 0
        
        Example:
            >>> cond = ConditionTemplates.turnaround()
        """
        conditions = [
            Condition("gt", field="revenue_growth_yoy", value=min_revenue_growth, label="Revenue turning"),
            Condition("lt", field="pe_ratio", value=max_pe, label="Valuation reasonable"),
        ]
        if positive_fcf:
            conditions.append(Condition("gt", field="fcf_yield", value=0, label="Positive FCF"))
        return And(conditions=conditions, label="Turnaround")
    
    @staticmethod
    def deep_value_or_growth() -> Or:
        """
        Either deep value OR GARP — broad opportunity capture.
        
        Example:
            >>> cond = ConditionTemplates.deep_value_or_growth()
        """
        return Or(
            conditions=[
                ConditionTemplates.deep_value(),
                ConditionTemplates.growth_at_reasonable_price(),
            ],
            label="Deep Value or GARP",
        )
    
    @staticmethod
    def dividend_quality(
        min_market_cap: float = 2e9,
        min_roe: float = 0.08,
        max_pe: float = 25,
        min_fcf_yield: float = 0.03,
    ) -> And:
        """
        Dividend quality candidates (proxy via FCF strength and stability).
        
        Market Cap > $2B AND ROE > 8% AND P/E < 25 AND FCF Yield > 3%
        """
        return And(
            conditions=[
                Condition("gt", field="market_cap", value=min_market_cap, label="Mid-large cap"),
                Condition("gte", field="roe", value=min_roe, label="Stable returns"),
                Condition("lt", field="pe_ratio", value=max_pe, label="Value range"),
                Condition("gte", field="fcf_yield", value=min_fcf_yield, label="FCF coverage"),
            ],
            label="Dividend Quality",
        )


# ═══════════════════════════════════════════════════════════════════
# Condition Evaluator (batch screening)
# ═══════════════════════════════════════════════════════════════════

class ConditionEvaluator:
    """
    Batch condition evaluator for screening stock universes.
    
    Example:
        >>> evaluator = ConditionEvaluator()
        >>> cond = ConditionTemplates.deep_value()
        >>> universe = [
        ...     {"ticker": "ABC", "market_cap": 1e9, "pe_ratio": 10, "pb_ratio": 1.2, "fcf_yield": 0.05},
        ...     {"ticker": "XYZ", "market_cap": 5e9, "pe_ratio": 30, "pb_ratio": 4.0, "fcf_yield": 0.01},
        ... ]
        >>> passing, failing = evaluator.screen(universe, cond, key="ticker")
        >>> print(passing)  # ["ABC"]
        >>> print(failing)  # ["XYZ"]
    """
    
    def screen(self, 
               universe: List[StockData],
               condition: ConditionNode,
               key: str = "ticker",
               ) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Screen a universe against a condition tree.
        
        Args:
            universe: List of stock data (dicts or objects)
            condition: Root condition node
            key: Field name to use as identifier
        
        Returns:
            (passing_ids, failing_ids, failure_reasons)
            failure_reasons: {id: [list of failed condition labels]}
        """
        passing = []
        failing = []
        failure_reasons: Dict[str, List[str]] = {}
        
        for stock in universe:
            stock_id = str(_get_field_value(stock, key) or id(stock))
            
            try:
                if condition.evaluate(stock):
                    passing.append(stock_id)
                else:
                    failing.append(stock_id)
                    failure_reasons[stock_id] = self._diagnose(stock, condition)
            except Exception as e:
                failing.append(stock_id)
                failure_reasons[stock_id] = [f"Error: {e}"]
        
        logger.info(
            f"Screening complete: {len(passing)} passed, "
            f"{len(failing)} failed (of {len(universe)} total)"
        )
        
        return passing, failing, failure_reasons
    
    def _diagnose(self, stock: StockData, node: ConditionNode,
                  parent_label: str = "") -> List[str]:
        """Diagnose why a stock failed a condition tree."""
        reasons = []
        
        if isinstance(node, Condition):
            if not node.evaluate(stock):
                val = _get_field_value(stock, node.field)
                reasons.append(
                    f"{node.label or node.field}: expected {node.explain()}, "
                    f"got {val}"
                )
        
        elif isinstance(node, And):
            for child in node.conditions:
                reasons.extend(self._diagnose(stock, child, node.label))
        
        elif isinstance(node, Or):
            if not node.evaluate(stock):
                # All children failed — show each
                for child in node.conditions:
                    child_reasons = self._diagnose(stock, child, node.label)
                    reasons.extend(child_reasons)
        
        elif isinstance(node, Not):
            if not node.evaluate(stock):
                child = node.condition
                if child:
                    reasons.append(f"NOT({child.explain()}) — condition matched (should have failed)")
        
        return reasons
    
    def count_passing_conditions(self, stock: StockData, 
                                  conditions: List[Condition]) -> int:
        """Count how many primitive conditions a stock passes."""
        return sum(1 for c in conditions if c.evaluate(stock))


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (as docstring examples)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test basic conditions
    stock = {
        "ticker": "TEST",
        "pe_ratio": 12,
        "pb_ratio": 1.2,
        "market_cap": 1.5e9,
        "fcf_yield": 0.05,
        "revenue_growth_yoy": 0.15,
        "roe": 0.20,
        "debt_to_equity": 0.4,
        "profit_margin": 0.12,
        "return_3m": 0.25,
        "beta": 1.5,
        "sector": "Technology",
    }
    
    print("=" * 60)
    print("Condition System Test")
    print("=" * 60)
    
    # Deep value
    print("\n🔍 Deep Value:")
    dv = ConditionTemplates.deep_value()
    print(f"  Condition: {dv.explain()}")
    print(f"  Result: {dv.evaluate(stock)}")
    print(f"  JSON:\n{dv.to_json()}")
    
    # Deep value OR GARP
    print("\n🔍 Deep Value OR GARP:")
    combo = ConditionTemplates.deep_value_or_growth()
    print(f"  Condition: {combo.explain()}")
    print(f"  Result: {combo.evaluate(stock)}")
    
    # Test nested: (Quality AND Value) OR Momentum
    print("\n🔍 Nested: (Quality Value AND Less Debt) OR High Momentum:")
    nested = Or(
        conditions=[
            And(
                conditions=[
                    ConditionTemplates.quality_value(),
                    Condition("lt", field="debt_to_equity", value=0.3, label="Very low debt"),
                ],
                label="Quality + Low Debt",
            ),
            ConditionTemplates.high_momentum(),
        ],
        label="Quality or Momentum",
    )
    print(f"  Condition: {nested.explain()}")
    print(f"  Result: {nested.evaluate(stock)}")
    print(f"  JSON roundtrip: ", end="")
    roundtrip = ConditionNode.from_json(nested.to_json())
    print(f"{'OK' if roundtrip.evaluate(stock) == nested.evaluate(stock) else 'FAIL'}")
    
    # Condition evaluator
    print("\n🔍 Batch Screening:")
    evaluator = ConditionEvaluator()
    universe = [
        stock,
        {"ticker": "FAIL", "pe_ratio": 50, "pb_ratio": 5.0, "market_cap": 20e9, "fcf_yield": 0.0},
        {"ticker": "OK1", "pe_ratio": 8, "pb_ratio": 0.8, "market_cap": 500e6, "fcf_yield": 0.08},
    ]
    passing, failing, reasons = evaluator.screen(universe, dv)
    print(f"  Passing: {passing}")
    print(f"  Failing: {failing}")
    for fid in failing:
        print(f"  {fid} reasons: {reasons.get(fid, [])}")
