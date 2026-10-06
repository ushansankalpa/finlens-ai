# Live execution audit, 2026-10-07 Asia/Colombo

These files contain actual provider-driven attempts, not fixture traces. They are retained to make failures visible. No complete Task 3 submission is claimed.

- `61ce...`: initial run fetched prices but exceeded the provider request-token limit. Model context now omits OHLCV tables and redundant evidence copies while full observations remain saved.
- `344ac6...`: failed run evaluated model-invented headlines before a provenance guard existed. Those sentiment results must not be used. The dispatcher now accepts only exact IDs/titles from successful news retrieval and uses the stored source records; a regression check rejects fabricated records.
- `1caf881259834b5c9f64c0267173a11f`: preliminary two-agent report completed with price/volatility observations and typed clarification. Analyst attempts at Writer-only tools were rejected and logged. No news was retrieved. Claims about historical valuation and general macro risks need review against actual observations. This is a debugging artifact, not an accepted grounded report.
- `0392d422d0e6475fb4e31561b2d43c4e`: after role instructions were clarified, price/volatility succeeded; provider HTTP 429 exhausted bounded retries before completion.

Next validation: rerun single/multi modes when account quota permits; inspect source relevance and every report claim, record news/sentiment and critique incorporation, then run memory/cache and controlled-failure notebook stages. Keep original failure records separate from the accepted run when preparing submission evidence.
