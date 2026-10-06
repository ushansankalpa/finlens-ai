from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from shared.io import json_safe
from task1_financial.config import TRADING_DAYS
from task1_financial.data import build_summary, fetch_market, validate_ticker
from task1_financial.indicators import INDICATOR_COLUMNS, compute_indicators
from task1_financial.news import get_news as fetch_news
from task1_financial.pipeline import aggregate_sentiment, analyze_headlines
from task1_financial.schemas import Headline, StrictModel


class PriceArgs(StrictModel):
    ticker: str
    period: str = Field(pattern=r"^(1y|2y|3y|5y)$")


class NewsArgs(StrictModel):
    ticker: str
    n: int = Field(ge=1, le=50)


class VolatilityArgs(StrictModel):
    ticker: str
    window: int = Field(ge=2, le=500)


class SentimentArgs(StrictModel):
    headlines: list[Headline] = Field(min_length=1, max_length=50)


class SearchArgs(StrictModel):
    query: str = Field(min_length=3, max_length=500)


def annualized_volatility(close: pd.Series, window: int):
    returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna().tail(window)
    if len(returns) < window:
        raise ValueError(f"Need {window} valid returns; received {len(returns)}")
    return {"annualized_volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "window": window, "sample_count": len(returns), "return_type": "log", "ddof": 1,
            "trading_days_per_year": TRADING_DAYS, "first_date": returns.index[0].isoformat(),
            "last_date": returns.index[-1].isoformat()}


class FinancialTools:
    def __init__(self, client):
        self.client = client
        self.market_cache = {}

    def _market(self, ticker, period="3y"):
        ticker = validate_ticker(ticker)
        key = (ticker, period)
        if key not in self.market_cache:
            data, info, warnings = fetch_market(ticker, period=period)
            self.market_cache[key] = (compute_indicators(data), info, warnings)
        return self.market_cache[key]

    def get_price_data(self, ticker, period):
        frame, info, warnings = self._market(ticker, period)
        fields = ["totalRevenue", "profitMargins", "debtToEquity", "currentRatio", "freeCashflow", "totalCash", "totalDebt"]
        return {"summary": build_summary(ticker, frame, info, warnings).model_dump(),
                "indicators": json_safe(frame.iloc[-1][INDICATOR_COLUMNS].to_dict()),
                "fundamentals": {k: info.get(k) for k in fields},
                "fundamentals_note": "Provider snapshot; fiscal periods may differ or be unavailable",
                "ohlcv": json_safe([{**row, "date": str(index)} for index, row in
                                      frame.tail(20).to_dict(orient="index").items()]),
                "row_count": len(frame), "source": "yfinance adjusted OHLC"}

    def get_news(self, ticker, n):
        headlines, warnings = fetch_news(ticker, n)
        if not headlines:
            raise ValueError("No recent news returned; try analyst commentary search")
        return {"headlines": [h.model_dump() for h in headlines], "warnings": warnings}

    def calculate_volatility(self, ticker, window):
        frame, _, _ = self._market(ticker)
        return annualized_volatility(frame.Close, window)

    def llm_sentiment(self, headlines):
        validated = [Headline.model_validate(h) for h in headlines]
        results, warnings = analyze_headlines(validated, self.client)
        if not results:
            raise ValueError("No validated sentiment results")
        return {"score": aggregate_sentiment(results), "sentiments": json_safe(results),
                "warnings": warnings, "validated_count": len(results)}

    def web_search(self, query):
        from duckduckgo_search import DDGS
        results = list(DDGS(timeout=20).text(query, max_results=5))
        if not results:
            raise ValueError("No search results; try a more specific query or news source")
        return {"results": results, "limitations": "Search snippets require source verification for additional claims"}


ARG_SCHEMAS = {"get_price_data": PriceArgs, "get_news": NewsArgs,
               "calculate_volatility": VolatilityArgs, "llm_sentiment": SentimentArgs,
               "web_search": SearchArgs}
ROLE_TOOLS = {"researcher": set(ARG_SCHEMAS),
              "analyst": {"get_price_data", "calculate_volatility", "llm_sentiment"},
              "writer": {"web_search", "get_news"}}
