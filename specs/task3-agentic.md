# Task 3 specification

Status: initial implementation and fixture control-flow checks present. Assessment requirements: PDF pages 10-12. Live agent execution awaits credentials; see ../EVIDENCE.md.

## Goal and output contract

Input: "Analyse the current financial health and market sentiment of [TICKER]. Identify the top three risks to its share price over the next 90 days and suggest one data-driven hedge strategy."

Output: validated `ResearchReport` with `financial_health_summary`, exactly three `risks` each with supporting evidence, and one `hedge_strategy_recommendation`. Include ticker, as-of time, provenance, warnings, and available quantitative inputs. The 90-day horizon is specified in reasoning; do not invent forecasts or current options prices.

Chosen framework proposal: LangGraph for explicit state, routing, and inspectable tool/agent transitions. Verify installed API and pin a working version before implementation. Reuse Task 1 numerical, data, schema and inference utilities.

## Tools and contracts

| Tool | Inputs | Output |
|---|---|---|
| get_price_data | ticker, period | Clean OHLCV/compact statistics, indicators, provenance; include available fundamentals metadata for financial-health reasoning |
| get_news | ticker, n | Typed headline collection with source/time metadata |
| calculate_volatility | ticker, window | Annualized standard deviation of daily log returns using sqrt(252), stated sample convention, sample count and date range |
| llm_sentiment | headlines | Validated headline sentiments, heuristic aggregate and failure counts |
| web_search | query | Search result titles, snippets, URLs, retrieval time using assessment-required duckduckgo-search |

All tools return a structured success/error envelope and never silently fabricate fallback data. Bound timeouts/retries. Daily prices alone do not prove balance-sheet health: use available fundamentals or explicitly limit that section. Search snippets are evidence leads; claims beyond their content need source verification.

## Single-agent autonomy

First implement a research agent with all five tools. A model action chooses a permitted tool plus schema-validated arguments or a final report. Each observation updates session state before the next choice. A scheduler may enforce limits, but must not hardcode tool ordering. Record a live call -> observation -> changed next action. Demonstrate a failure causing a genuinely different available approach rather than just stopping.

## Two-agent roles and typed coordination

- Analyst tools: only get_price_data, calculate_volatility, llm_sentiment.
- Writer tools: only web_search, get_news.
- Enforce restrictions using separate tool registries and dispatcher checks, not just role prompts.
- `DataBrief` contains ticker, timestamps, quantitative observations, sentiment if available, typed evidence references and limitations.
- `HeadlineBundle` allows Writer -> Analyst handoff for sentiment because Analyst cannot fetch news.
- `ClarificationRequest` names one missing/specific observation and why it matters; `ClarificationResponse` contains the requested data or explicit unavailable status with evidence.
- State stores typed handoffs; notebook displays the agent messages and serialized payloads.

For the assessed demonstration, require a clarification checkpoint before finalization: Writer generates one evidence-specific question, Analyst answers through its permitted tools or context, Writer incorporates the answer. The checkpoint guarantees the required loop is exercised; it must not prescribe the agents' tool order or question content. Cap clarification count and total tool steps to avoid an infinite loop.

## Memory and observability

- Session state retains tool observations and evidence. Demonstrate a follow-up asking for a previously retrieved number; compare tool-call counts before/after to prove no refetch.
- Save the final report under `cache/{ticker}/{date}.json` with config/schema version. On a subsequent same-ticker/date/config run, load the report and show that tools are not called. Explicit refresh overrides cache; stale/incompatible entries are not reused.
- Commit actual `logs/agent_trace.jsonl`. Every call, including failure, records timestamp, run_id, agent, tool_name, input arguments, output summary truncated to 200 characters, status, and duration_ms measured with a monotonic clock.
- Save a separate complete agent/handoff trace where useful and display it in notebook output. A 200-character tool log alone is insufficient evidence for full message/handoff review.
- Redact credentials and avoid storing authentication headers. Preserve public-source attribution in the report.

## Acceptance criteria

| ID | Requirement | Gate |
|---|---|---|
| T3-01 | Five working tools | Live examples with correct typed returns; quantitative fixture for volatility |
| T3-02 | Autonomous single-agent research | Visible model-selected call/observe/replan sequence and grounded final report |
| T3-03 | Graceful tool failure | Controlled failed tool followed by alternative and truthful warnings |
| T3-04 | Restricted two-agent roles | Unauthorized dispatch rejected; live trace shows permitted usage |
| T3-05 | Typed handoff | Validated DataBrief/HeadlineBundle payloads printed in notebook |
| T3-06 | Executed critique loop | Specific request, answer, and visible incorporation before final report |
| T3-07 | Fully automated initial-query-to-report run | No intermediate manual input required |
| T3-08 | Short-term memory | Follow-up reuses observation with zero additional retrieval calls |
| T3-09 | Persistent memory | Second compatible run loads saved JSON and does not rerun tools |
| T3-10 | Every tool call logged | Actual JSONL entries include inputs, <=200-character output, durations, errors |
| T3-11 | Final report and Colab evidence | Three sections, exactly three supported risks, one justified hedge, complete notebook trace |

Use controlled failure injection to verify fallback behavior and label that demonstration explicitly. The principal assessment run must use real data/LLM tools. Keep a finite step budget and explicit partial-failure outcome. Skip optional observability dashboard until all required evidence passes.
