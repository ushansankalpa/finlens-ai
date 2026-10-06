from __future__ import annotations

import argparse
import logging
from pathlib import Path

from shared.io import json_safe, utc_now, write_json
from shared.llm import JSONClient, LLMError
from task1_financial.config import MIN_HEADLINES
from task1_financial.data import DataError, build_summary, fetch_market, validate_ticker
from task1_financial.indicators import INDICATOR_COLUMNS, compute_indicators
from task1_financial.news import get_news
from task1_financial.prompts import SENTIMENT_SYSTEM, SIGNAL_SYSTEM
from task1_financial.schemas import HeadlineSentiment, ResearchResult, Signal


def analyze_headlines(headlines, client):
    results, warnings = {}, []
    for headline in headlines:
        try:
            result = client.complete(HeadlineSentiment, SENTIMENT_SYSTEM, {"headline": headline.title})
            if result.headline.strip() != headline.title.strip():
                raise LLMError("Returned headline does not match input")
            results[headline.id] = result
        except LLMError as error:
            warnings.append(f"Sentiment unavailable for {headline.id}: {error}")
    return results, warnings


def aggregate_sentiment(results) -> float | None:
    values = list(results.values()) if isinstance(results, dict) else list(results)
    total = sum(item.confidence for item in values)
    if total == 0:
        return None
    labels = {"positive": 1, "neutral": 0, "negative": -1}
    return sum(labels[item.sentiment] * item.confidence for item in values) / total


def run(ticker: str = "AAPL", output_dir="task1_financial/outputs", client=None):
    ticker = validate_ticker(ticker)
    client = client or JSONClient()
    data, info, warnings = fetch_market(ticker)
    frame = compute_indicators(data)
    summary = build_summary(ticker, frame, info, warnings)
    indicators = json_safe(frame.iloc[-1][INDICATOR_COLUMNS].to_dict())
    headlines, news_warnings = get_news(ticker, MIN_HEADLINES, info.get("shortName"))
    warnings = summary.warnings + news_warnings
    sentiments, sentiment_warnings = analyze_headlines(headlines, client)
    warnings.extend(sentiment_warnings)
    aggregate = aggregate_sentiment(sentiments)
    signal = None
    if all(value is not None for value in indicators.values()):
        try:
            signal = client.complete(Signal, SIGNAL_SYSTEM,
                                     {"summary": summary.model_dump(), "indicators": indicators,
                                      "aggregate_sentiment": aggregate})
            if not set(signal.evidence_identifiers).issubset(indicators):
                raise LLMError("Signal contains unknown indicator evidence identifiers")
        except LLMError as error:
            warnings.append(f"Signal unavailable: {error}")
            signal = None
    else:
        warnings.append("Signal unavailable: essential indicators missing")
    result = ResearchResult(summary=summary, indicators=indicators, headlines=headlines,
                            sentiments=sentiments, aggregate_sentiment=aggregate, signal=signal,
                            warnings=warnings, metadata={"retrieved_at": utc_now(), "provider": "Groq",
                            "model": client.model, "source": "yfinance and RSS",
                            "first_bar": frame.index[0].isoformat(), "last_bar": frame.index[-1].isoformat(),
                            "row_count": len(frame), "inference_calls": client.calls,
                            "valid_sentiments": len(sentiments), "failed_sentiments": len(headlines)-len(sentiments)})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{ticker}_indicators.csv", index_label="Date")
    write_json(directory / f"{ticker}_research.json", result)
    from task1_financial.report import render_report
    render_report(result, frame, directory)
    return result, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--output-dir", default="task1_financial/outputs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        result, _ = run(args.ticker, args.output_dir)
    except DataError as error:
        parser.exit(1, f"Data pipeline failed: {error}\n")
    print(result.model_dump_json(indent=2))
    if result.signal is None or len(result.headlines) < MIN_HEADLINES or len(result.sentiments) < len(result.headlines):
        parser.exit(2, "Partial execution: check report warnings; assessment evidence incomplete.\n")


if __name__ == "__main__":
    main()
