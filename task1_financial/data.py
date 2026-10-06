from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from shared.io import json_safe
from task1_financial.indicators import momentum
from task1_financial.schemas import MarketSummary

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    pass


def validate_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", ticker):
        raise DataError("Invalid ticker format")
    return ticker


def clean_ohlcv(raw: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, list[str]]:
    if raw is None or raw.empty:
        raise DataError("Market-data source returned no observations")
    data = raw.copy()
    warnings = []
    if isinstance(data.columns, pd.MultiIndex):
        matching = [level for level in range(data.columns.nlevels)
                    if ticker in data.columns.get_level_values(level)]
        if matching:
            data = data.xs(ticker, axis=1, level=matching[0])
        elif any(set(OHLCV).issubset(set(data.columns.get_level_values(level)))
                 for level in range(data.columns.nlevels)):
            other = [level for level in range(data.columns.nlevels)
                     if not set(OHLCV).issubset(set(data.columns.get_level_values(level)))]
            if any(len(data.columns.get_level_values(level).unique()) > 1 for level in other):
                raise DataError("Requested ticker absent from multi-ticker data")
            data.columns = data.columns.get_level_values(next(
                level for level in range(data.columns.nlevels)
                if set(OHLCV).issubset(set(data.columns.get_level_values(level)))))
    if not set(OHLCV).issubset(data.columns):
        raise DataError("OHLCV columns missing from source data")
    data = data[OHLCV].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    data.index = pd.to_datetime(data.index, errors="coerce", utc=True).tz_convert(None).normalize()
    data = data[~data.index.isna()]
    duplicate_count = int(data.index.duplicated().sum())
    if duplicate_count:
        warnings.append(f"Removed {duplicate_count} duplicate timestamps")
    data = data[~data.index.duplicated(keep="last")].sort_index()
    valid = (data[["Open", "High", "Low", "Close"]].notna().all(axis=1)
             & (data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
             & (data["High"] >= data[["Open", "Low", "Close"]].max(axis=1))
             & (data["Low"] <= data[["Open", "High", "Close"]].min(axis=1)))
    removed = int((~valid).sum())
    if removed:
        warnings.append(f"Dropped {removed} invalid/incomplete price rows; windows use remaining observations")
    data = data.loc[valid]
    if data["Volume"].isna().any() or (data["Volume"] < 0).any():
        warnings.append("Unavailable/negative volume preserved as missing")
        data.loc[data["Volume"] < 0, "Volume"] = np.nan
    if data.empty:
        raise DataError("No usable price observations after cleaning")
    return data, warnings


def fetch_market(ticker: str, now=None, period: str | None = None):
    ticker = validate_ticker(ticker)
    now = pd.Timestamp(now or datetime.now(timezone.utc)).tz_localize(None).normalize()
    # Exclude the current UTC calendar day, conservatively avoiding unfinished bars.
    args = {"period": period} if period else {"start": now - pd.DateOffset(years=3), "end": now}
    try:
        raw = yf.download(ticker, **args, auto_adjust=True, progress=False, threads=False, timeout=20)
    except Exception:
        raise DataError("Market-data fetch failed") from None
    data, warnings = clean_ohlcv(raw, ticker)
    data = data.loc[data.index < now]
    if data.empty:
        raise DataError("No completed daily bars available")
    if not period and data.index.min() > data.index.max() - pd.DateOffset(years=2):
        warnings.append("Returned history covers less than two years; assessment coverage requirement unmet")
    try:
        info = yf.Ticker(ticker).get_info() or {}
    except Exception:
        info = {}
        warnings.append("Company metadata unavailable")
    return data, json_safe(info), warnings


def build_summary(ticker: str, frame: pd.DataFrame, info: dict, warnings: list[str]) -> MarketSummary:
    latest = json_safe(frame.iloc[-1].to_dict())
    as_of = frame.index[-1]
    year_start = pd.Timestamp(year=as_of.year, month=1, day=1)
    baseline = frame.loc[frame.index < year_start, "Close"]
    ytd = latest["Close"] / float(baseline.iloc[-1]) - 1 if not baseline.empty else None
    recent = frame.loc[frame.index >= as_of.to_pydatetime() - timedelta(weeks=52)]
    local_warnings = list(warnings)
    pe = info.get("trailingPE")
    try:
        pe = float(pe) if pe is not None and np.isfinite(float(pe)) else None
    except (TypeError, ValueError):
        pe = None
    if pe is None:
        local_warnings.append("Trailing P/E unavailable")
    if ytd is None:
        local_warnings.append("YTD return unavailable: previous year-end baseline missing")
    return MarketSummary(ticker=ticker, as_of=as_of.isoformat(), currency=info.get("currency"),
                         latest_close=latest["Close"], high_52w=float(recent["High"].max()),
                         low_52w=float(recent["Low"].min()), pe_ratio=pe, ytd_return=ytd,
                         momentum_signal=momentum(latest), warnings=local_warnings)
