from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class BenchmarkPoint:
    dt: date
    value: float
    source_url: str


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    instrument_type: str
    sector: str
    region: str
    quality_score: float
    tradable: bool = True


@dataclass(frozen=True)
class OrderSuggestion:
    dt: date
    ticker: str
    side: str
    amount_clp: float
    quantity: float

