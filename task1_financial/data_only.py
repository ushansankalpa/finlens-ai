"""Real financial data evidence without requiring an inference credential."""
import argparse
from pathlib import Path

from shared.io import json_safe, utc_now, write_json
from task1_financial.data import DataError, build_summary, fetch_market
from task1_financial.indicators import INDICATOR_COLUMNS, compute_indicators
from task1_financial.news import get_news


def run(ticker="AAPL", output_dir="task1_financial/outputs"):
    frame, info, warnings = fetch_market(ticker)
    frame = compute_indicators(frame)
    summary = build_summary(ticker, frame, info, warnings)
    headlines, news_warnings = get_news(ticker, 10, info.get("shortName"))
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{ticker}_indicators.csv", index_label="Date")
    evidence = {"execution_stage": "live_data_only", "retrieved_at": utc_now(),
                "summary": summary.model_dump(), "indicators": json_safe(frame.iloc[-1][INDICATOR_COLUMNS].to_dict()),
                "headlines": [h.model_dump() for h in headlines], "warnings": summary.warnings + news_warnings,
                "first_bar": frame.index[0].isoformat(), "last_bar": frame.index[-1].isoformat(),
                "row_count": len(frame), "llm_execution": "not attempted"}
    write_json(directory / f"{ticker}_data_evidence.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    args = parser.parse_args()
    import json
    try:
        print(json.dumps(run(args.ticker), indent=2))
    except DataError as error:
        parser.exit(1, f"Data pipeline unavailable: {error}\n")
