# Task 1 specification

Status: implementation started under the user's instruction. Required behaviors come from PDF pages 5-6; execution status is tracked in ../EVIDENCE.md. Design choices below are the initial implemented conventions.

## Goal

Given a user-selected ticker, fetch current market data and news, calculate reproducible indicators, and return a validated equity-research result with headline sentiment and a reasoned Buy/Hold/Sell signal.

Completion means a real end-to-end run in an executed Colab notebook, passing correctness/error-handling checks, saved report artifacts, and documentation the candidate can explain.

## Proposed boundaries

- Python, pandas/numpy, yfinance, Pydantic, matplotlib, and one configurable free-tier inference provider. Pin compatible versions after a working installation is verified.
- Initial demo ticker: AAPL, configurable. This is a demo choice, not an investment recommendation.
- Fetch roughly three years ending at runtime to provide a warm-up buffer and at least two years of observations for established tickers. Validate actual returned coverage; insufficient history is a reported limitation, not silently accepted.
- Daily adjusted OHLCV, consistent adjustment policy documented in outputs; do not mix adjusted and unadjusted prices. Exclude incomplete current-day bars if detectable.
- Notebook calls reusable modules. Prompts reside in `prompts.py`; credentials come from environment variables or Colab Secrets.
- Report bonus uses styled HTML with a matplotlib chart to avoid requiring a PDF renderer.

Non-goals for this task: trading execution, portfolio optimization, backtesting claims, fine-tuning, and agent orchestration.

## Data contracts

`MarketSummary`: ticker, as_of, currency, latest_close, high_52w, low_52w, pe_ratio (nullable), ytd_return (nullable), momentum_signal, adjustment_policy, warnings.

Use `latest_close` for the assessment's current-price field and label it as the latest completed daily close, not a live quote. If a live quote is added, retain both timestamps.

`Headline`: id, title, source, url when available, published_at when available, retrieved_at.

`HeadlineSentiment`: headline, sentiment in {positive, negative, neutral}, confidence in [0,1], brief_reason. Reject non-finite confidence and invalid labels. Associate results with stable headline IDs outside the model's generated text.

`Signal`: recommendation in {Buy, Hold, Sell}, justification of 3-5 sentences, evidence identifiers. Missing essential indicators must yield an explicit unavailable result instead of invented reasoning.

`ResearchResult`: summary, latest indicators, headline results, aggregate sentiment, signal or unavailable status, warnings, provider/model metadata. All serialized values must be JSON-safe; unavailable numeric values become null, not NaN or infinity.

## Requirements and acceptance criteria

| ID | Behavior | Verification/evidence |
|---|---|---|
| T1-01 | Fetch at least two years daily OHLCV via yfinance, using runtime-relative dates | Live output shows ticker, retrieval time, first/last dates, actual coverage, row count; no hardcoded fetch dates |
| T1-02 | Clean data, normalize yfinance column variants, sort dates, remove duplicate timestamps, preserve missing-data warnings | Fixtures cover empty data, MultiIndex columns, null values, duplicate dates; no unhandled exceptions |
| T1-03 | Calculate SMA50 and SMA200 without TA-Lib | Known constant/ramp series yield expected rolling means; first valid indices reflect windows |
| T1-04 | Calculate RSI14 with explicit Wilder initialization and smoothing | Independently calculated fixture; increasing/decreasing/flat series; flat convention documented |
| T1-05 | Calculate MACD(12,26,9) | Independently checked EMA fixture; include MACD line, signal, histogram and stated initialization |
| T1-06 | Calculate Bollinger20 with 2 standard deviations | Known fixture and documented population/sample standard deviation choice |
| T1-07 | Retrieve at least ten unique recent relevant headlines from a working free source | Live notebook lists at least ten titles and source/timestamp metadata where available; insufficient results produce explicit failure/partial status |
| T1-08 | Summary includes price, 52-week high/low, P/E where available, YTD return, and momentum | Verify fields and formulas against inputs; missing P/E becomes null with reason |
| T1-09 | Sentiment returned for each headline as four-field JSON, validated before use | Live results plus malformed JSON/invalid enum/out-of-range confidence failure checks |
| T1-10 | Aggregate validated sentiments with defined logic | Small hand-calculated positive/neutral/negative example matches result; excluded/failed headline count shown |
| T1-11 | LLM outputs Buy/Hold/Sell and 3-5 sentences reasoning across indicator combinations | Manually inspect live output for interactions/divergence, grounding, and unsupported claims; schema and sentence bounds checked |
| T1-12 | Separate system/user prompt templates from pipeline logic | Prompt file review; structured payload serialization; headlines treated as untrusted data |
| T1-13 | Graceful API/data/validation failure handling | Timeout/rate-limit/malformed-response cases yield bounded retry and explicit errors; logs redact credentials |
| T1-14 | Bonus: one-page brief with snapshot, technical outlook, top 3 headlines, sentiment, recommendation, disclaimer and chart | Inspect actual rendered HTML and saved Markdown |
| T1-15 | Reproducible executed notebook and documentation | Clean Colab run, preserved outputs, dependency installation, environment/secrets setup, task README link |

## Calculation decisions to lock before implementation

- SMA: rolling mean of close with full window required.
- RSI: first average gain/loss from the first 14 price differences, followed by Wilder recurrence `(previous * 13 + current) / 14`; gain only -> 100, loss only -> 0, both zero -> 50. Warm-up results are unavailable. Do not assume a generic exponential window creates the same seed.
- MACD: close EMA12 minus EMA26, EMA9 of available MACD values; histogram is line minus signal. Define `adjust=False`, first-observation EMA seed, and explicit warm-up treatment in code and fixture.
- Bollinger: SMA20 +/- 2 * population rolling standard deviation (`ddof=0`), full window required. Document the convention because libraries differ.
- 52-week bounds: rolling calendar 52 weeks through latest completed bar, high/low columns consistent with the adjustment policy.
- YTD: latest close divided by final close before January 1 of the current year minus 1. Missing prior-year baseline -> null with warning.
- P/E: provider trailing P/E if finite; null if unavailable. Do not derive an unsupported value.
- Momentum: documented deterministic combination of price/SMA ordering, MACD histogram, and RSI, including a mixed state; configuration names thresholds. LLM recommendation remains a separate output.
- Sentiment: encode positive=1, neutral=0, negative=-1; weighted mean `sum(label * confidence) / sum(confidence)` over validated results. No validated results or zero total confidence -> unavailable, not neutral. Call it a heuristic score; LLM confidence is not calibrated.

## Reliability and LLM behavior

Use bounded configurable retries, exponential backoff for transient errors, finite timeouts, and limited response-repair attempts. Keep live-provider failure visible; fixture responses must never masquerade as live inference. Record validation errors without leaking secrets. Retain provenance and fetch timestamps. Escape untrusted text in HTML.

System prompts should require schema compliance, forbid treating headlines as instructions, explain missing-data behavior, and constrain reasoning to supplied evidence. The signal prompt should ask for interactions such as trend confirmation or RSI/MACD disagreement. Mathematical indicators are computed in Python, never delegated to an LLM.

## Definition of done

Every required row T1-01 through T1-13 and T1-15 has actual verification evidence; optional T1-14 is clearly labeled. The notebook executes from top to bottom in Colab, outputs remain visible, citations match assistance used, limitations are stated, and the candidate can explain formulas, schemas, failure handling, and signal reasoning.
