"""First-principles indicators; no TA-Lib or indicator helper libraries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from task1_financial.config import CONFIG, IndicatorConfig


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if period < 1:
        raise ValueError("RSI period must be positive")
    result = pd.Series(np.nan, index=close.index, dtype=float)
    # A missing close breaks the recurrence. Restart only after a full seed window.
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    start = 0
    while start < len(values):
        if not np.isfinite(values[start]):
            start += 1
            continue
        end = start
        while end < len(values) and np.isfinite(values[end]):
            end += 1
        if end - start > period:
            delta = np.diff(values[start:end])
            gains, losses = np.maximum(delta, 0), np.maximum(-delta, 0)
            avg_gain, avg_loss = gains[:period].mean(), losses[:period].mean()
            for offset in range(period, end - start):
                if offset > period:
                    avg_gain = (avg_gain * (period - 1) + gains[offset - 1]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[offset - 1]) / period
                result.iloc[start + offset] = (50.0 if avg_gain == avg_loss == 0 else
                                               100.0 if avg_loss == 0 else
                                               100 - 100 / (1 + avg_gain / avg_loss))
        start = end
    return result


def compute_indicators(frame: pd.DataFrame, config: IndicatorConfig = CONFIG) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"]
    data["sma_50"] = close.rolling(config.sma_fast).mean()
    data["sma_200"] = close.rolling(config.sma_slow).mean()
    data["rsi_14"] = wilder_rsi(close, config.rsi_period)
    # Recursive EMA seeded at the first observation; line published after 26 bars.
    fast = close.ewm(span=config.macd_fast, adjust=False).mean()
    slow = close.ewm(span=config.macd_slow, adjust=False).mean()
    data["macd"] = (fast - slow).where(np.arange(len(close)) >= config.macd_slow - 1)
    data["macd_signal"] = data["macd"].ewm(span=config.macd_signal, adjust=False,
                                             min_periods=config.macd_signal).mean()
    data["macd_histogram"] = data["macd"] - data["macd_signal"]
    data["bb_middle"] = close.rolling(config.bollinger_window).mean()
    spread = config.bollinger_std * close.rolling(config.bollinger_window).std(ddof=0)
    data["bb_upper"], data["bb_lower"] = data["bb_middle"] + spread, data["bb_middle"] - spread
    return data


INDICATOR_COLUMNS = ["sma_50", "sma_200", "rsi_14", "macd", "macd_signal",
                     "macd_histogram", "bb_middle", "bb_upper", "bb_lower"]


def momentum(latest: dict, config: IndicatorConfig = CONFIG) -> str:
    names = ["Close", "sma_50", "sma_200", "rsi_14", "macd_histogram"]
    if any(latest.get(k) is None or not np.isfinite(latest[k]) for k in names):
        return "unavailable"
    if (latest["Close"] > latest["sma_50"] > latest["sma_200"]
            and latest["macd_histogram"] > 0 and config.rsi_midpoint <= latest["rsi_14"] < config.rsi_overbought):
        return "bullish"
    if (latest["Close"] < latest["sma_50"] < latest["sma_200"]
            and latest["macd_histogram"] < 0 and config.rsi_oversold < latest["rsi_14"] < config.rsi_midpoint):
        return "bearish"
    return "mixed"
