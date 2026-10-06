# Acceptance and execution ledger

Updated 2026-10-07 (Asia/Colombo). This records actual execution separately from implemented code. No complete assessment submission is claimed yet.

## Evidence currently available

- `task1_financial/outputs/AAPL_data_evidence.json`: actual yfinance/RSS retrieval, timestamp, 751 bars from 2023-10-06 through 2026-10-05, ten recent headline records, required summary/indicators. Stage is explicitly `live_data_only`.
- `task1_financial/outputs/AAPL_indicators.csv`: actual adjusted OHLCV and calculated indicator history.
- `task1_financial/outputs/AAPL_research.json`: live Groq-backed report, ten validated sentiments, combined-indicator three-sentence Hold recommendation, zero warnings. Retrieval timestamp 2026-10-06T20:43:33Z corresponds to 2026-10-07 02:13 in Asia/Colombo. This is a research demonstration, not a trading recommendation to the candidate.
- `task1_financial/assessment.ipynb`: locally executed with actual live inference outputs retained. Its preliminary data cell explicitly reuses the dated data-only snapshot; the inference cell fetches fresh data.
- `task1_financial/outputs/AAPL_brief.html`, Markdown, and chart: rendered from the actual complete research result. Chart visually inspected; final personal report review remains.
- `validation/task1_initial_partial.json`: preserved initial live result rejected for unsupported signal evidence fields. The schema now enumerates allowed indicator identifiers so validation/repair catches this before use.
- `task3_agentic/assessment.ipynb`: locally executed fixture checks and explicit NOT EXECUTED statements for live stages. This is not live-agent research evidence.
- `validation/test-results.xml`: 37 passing local numerical, schema, transport, control-flow, policy, split-leakage, masking, dotenv and headline-provenance checks. Fixture-only agent logs use temporary test directories; none is represented as the required assessment trace.
- `task3_agentic/logs/multi_research.json`: preliminary live two-agent report with typed clarification and actual price/volatility observations. It did not obtain news, and its narrative requires grounding review; it is not accepted as the complete Task 3 result. A subsequent run with clearer role instructions stopped on HTTP 429.
- `task3_agentic/logs/agent_trace.jsonl` and per-run message files retain actual successful and failed runs, including rejected role permissions. An earlier failed run evaluated invented headlines before the provenance guard was added; those outputs are not verified evidence. See `task3_agentic/logs/RUN_AUDIT.md`.
- `task2_genai/outputs/tokenizer_check.json`: real Qwen tokenizer/chat-template check passed with complete assistant JSON preserved and prompt tokens masked. This is tokenizer-only evidence, not training.
- All three notebook JSON structures and code-cell syntax validated.
- Source/notebook scan found no Groq/HF credential patterns. No Git history exists yet to inspect.

## Requirement status

| Requirements | Status | Evidence / remaining work |
|---|---|---|
| T1-01 to T1-08 | Live data demonstrated; numerical checks verified | Data evidence, CSV, financial tests; inspect headline relevance and indicator conventions personally |
| T1-09 to T1-13 | Live inference demonstrated; validation/retry paths tested | Ten sentiments and grounded combined-indicator reasoning saved; candidate review remains |
| T1-14 | Complete inference-backed report rendered | HTML/Markdown/chart saved; inspect final layout personally |
| T1-15 | Local notebook fully executed with live outputs | Run in Colab and add real direct link before final submission |
| T2-01 | Domain contract and policy code implemented | Personally review use-case examples |
| T2-02 to T2-03 | Generator/oracle/split/diversity code implemented | Actual >=100 teacher examples, actual diversity report and split pending |
| T2-04 to T2-06 | Training and merge code implemented | Actual GPU NF4 smoke, loss history, decreasing validation loss, merge/reload and saved weights pending |
| T2-07 to T2-11 | Evaluation/review/notebook code implemented | Actual base/fine-tuned comparison, >=10 personal labels, two analysis paragraphs, accessible model pending |
| T3-01 | Five tool implementations present | Live end-to-end tool coverage pending; numerical volatility check verified |
| T3-02 to T3-07 | Fixture-tested; preliminary live multi-agent execution recorded | Price/volatility and typed critique executed; news/sentiment coverage and report grounding still need live verification |
| T3-08 to T3-10 | Memory/cache/logging fixture-tested | Live follow-up/cache evidence and committed actual agent_trace.jsonl pending |
| T3-11 | Notebook authored and locally exercised with live stages skipped | Complete live report and Colab output trace pending |
| SUB-01 to SUB-09 | Not complete | Public repo/name/links, full execution artifacts, finished reflection, personal video and access checks pending |

## Blockers and next actions

1. Groq key is configured privately in the root .env and the real inference smoke check passed. Colab still needs its own GROQ_API_KEY Secret; .env is never copied into the bundle.
2. Candidate confirmed a Colab GPU is available. Configure Colab Secrets, upload Task 2 notebook and code bundle, and run feasibility checks before longer jobs.
3. Task 1 ran locally with live outputs. Complete Task 3 live notebook verification when Groq quota permits; inspect outputs and retain all real traces.
4. Execute teacher generation, training, export and evaluation. Candidate performs manual labels and qualitative analysis. No training improvement can be promised beforehand.
5. Publish reviewed artifacts and personally record the mandatory video. The final repository name still needs the candidate name.

No optional report/agent/training bonus is claimed until its actual requirements have passed.
