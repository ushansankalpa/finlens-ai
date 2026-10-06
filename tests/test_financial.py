import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from shared.io import json_safe
from task1_financial.data import DataError, build_summary, clean_ohlcv
from task1_financial.indicators import compute_indicators, wilder_rsi
from task1_financial.pipeline import aggregate_sentiment
from task1_financial.schemas import HeadlineSentiment, Signal


def price_frame(close, dates=None):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": np.full(len(close), 1000)},
                        index=dates if dates is not None else pd.bdate_range("2023-01-02", periods=len(close)))


def test_rolling_means_and_population_bands():
    frame = compute_indicators(price_frame(np.arange(1, 251)))
    assert frame.sma_50.iloc[49] == pytest.approx(25.5)
    assert frame.sma_200.iloc[199] == pytest.approx(100.5)
    assert frame.sma_200.iloc[:199].isna().all()
    expected_std = math.sqrt((20 ** 2 - 1) / 12)
    assert frame.bb_upper.iloc[19] == pytest.approx(10.5 + 2 * expected_std)
    assert frame.bb_lower.iloc[19] == pytest.approx(10.5 - 2 * expected_std)


def test_wilder_seed_and_recurrence():
    # Widely used Wilder worked example; expected values computed independently.
    close = pd.Series([44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,
                       45.84,46.08,45.89,46.03,45.61,46.28,46.28,46.00])
    rsi = wilder_rsi(close)
    assert rsi.iloc[:14].isna().all()
    assert rsi.iloc[14] == pytest.approx(70.464135, abs=1e-5)
    assert rsi.iloc[15] == pytest.approx(66.249619, abs=1e-5)


@pytest.mark.parametrize("values,expected", [(range(30),100), (range(30,0,-1),0), ([10]*30,50)])
def test_rsi_edges(values, expected):
    assert wilder_rsi(pd.Series(values, dtype=float)).iloc[-1] == expected


def test_rsi_gap_restarts_seed():
    values = pd.Series(list(range(20)) + [np.nan] + list(range(20)), dtype=float)
    output = wilder_rsi(values)
    assert output.iloc[20:35].isna().all()
    assert output.iloc[35] == 100


def test_macd_against_independent_recursive_loop():
    close = [100 + math.sin(i / 5) * 10 for i in range(80)]
    output = compute_indicators(price_frame(close))
    fast = slow = close[0]
    signal = None
    line_values = []
    for i, value in enumerate(close):
        if i:
            fast = fast + (value - fast) * 2 / 13
            slow = slow + (value - slow) * 2 / 27
        if i >= 25:
            line = fast - slow
            line_values.append(line)
            signal = line if signal is None else signal + (line - signal) * 2 / 10
    assert output.macd.iloc[-1] == pytest.approx(line_values[-1])
    assert output.macd_signal.iloc[-1] == pytest.approx(signal)
    assert output.macd_histogram.iloc[-1] == pytest.approx(line_values[-1] - signal)
    assert output.macd_signal.iloc[:33].isna().all()


def test_clean_multiindex_and_null_rows():
    raw = price_frame([100, 101, 102])
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]])
    raw.loc[raw.index[1], ("Close", "AAPL")] = np.nan
    cleaned, warnings = clean_ohlcv(raw, "AAPL")
    assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(cleaned) == 2
    assert any("invalid" in warning for warning in warnings)


def test_empty_data_explicit_error():
    with pytest.raises(DataError, match="no observations"):
        clean_ohlcv(pd.DataFrame(), "AAPL")


def test_ytd_and_missing_pe():
    dates = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-02-02"])
    frame = compute_indicators(price_frame([90, 100, 105, 110], dates))
    summary = build_summary("AAPL", frame, {}, [])
    assert summary.ytd_return == pytest.approx(0.1)
    assert summary.pe_ratio is None
    assert summary.high_52w == 111
    assert summary.momentum_signal == "unavailable"
    assert any("P/E" in w for w in summary.warnings)


def test_sentiment_aggregation_and_invalid_values():
    def item(label, confidence):
        return HeadlineSentiment(headline="Test", sentiment=label, confidence=confidence, brief_reason="Test")
    assert aggregate_sentiment([item("positive",.8), item("negative",.2), item("neutral",.5)]) == pytest.approx(.4)
    assert aggregate_sentiment([]) is None
    assert aggregate_sentiment([item("positive",0)]) is None
    for confidence in [-.1, 1.1, float("nan")]:
        with pytest.raises(ValidationError):
            item("positive", confidence)


def test_signal_sentence_bounds_and_json_safety():
    with pytest.raises(ValidationError):
        Signal(recommendation="Buy", justification="One sentence.", evidence_identifiers=["rsi_14","macd"])
    assert json_safe({"x": float("nan"), "y": np.float64(float("inf"))}) == {"x": None, "y": None}


def test_signal_rejects_unsupported_evidence_fields():
    text = "The moving average trend remains positive. MACD momentum conflicts with that trend. RSI supports a cautious outlook."
    with pytest.raises(ValidationError):
        Signal(recommendation="Hold",justification=text,evidence_identifiers=["sma_50","invented_indicator"])
    assert Signal(recommendation="Hold",justification=text,evidence_identifiers=["sma_50","macd_histogram"])
