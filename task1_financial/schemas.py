from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Headline(StrictModel):
    id: str
    title: str = Field(min_length=1)
    source: str
    url: str | None = None
    published_at: str | None = None
    retrieved_at: str


class HeadlineSentiment(StrictModel):
    headline: str = Field(min_length=1)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    brief_reason: str = Field(min_length=1, max_length=1000)


class Signal(StrictModel):
    recommendation: Literal["Buy", "Hold", "Sell"]
    justification: str = Field(min_length=1)
    evidence_identifiers: list[Literal["sma_50", "sma_200", "rsi_14", "macd", "macd_signal",
                                      "macd_histogram", "bb_middle", "bb_upper", "bb_lower"]] = Field(min_length=2)

    @field_validator("justification")
    @classmethod
    def sentence_count(cls, value):
        # Avoid treating decimal points as sentence boundaries.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", value.strip()) if s.strip()]
        if not 3 <= len(sentences) <= 5:
            raise ValueError("Justification must contain three to five sentences")
        return value


class MarketSummary(StrictModel):
    ticker: str
    as_of: str
    currency: str | None
    latest_close: float | None
    high_52w: float | None
    low_52w: float | None
    pe_ratio: float | None
    ytd_return: float | None
    momentum_signal: Literal["bullish", "bearish", "mixed", "unavailable"]
    adjustment_policy: str = "yfinance auto_adjust=True: adjusted OHLC"
    warnings: list[str]


class ResearchResult(StrictModel):
    summary: MarketSummary
    indicators: dict[str, float | None]
    headlines: list[Headline]
    sentiments: dict[str, HeadlineSentiment]
    aggregate_sentiment: float | None
    signal: Signal | None
    warnings: list[str]
    metadata: dict
