# Assessment requirements and evidence

Source: user-provided assessment email and local `CDAZZDEV_Senior_MLE_Assessment_2026.pdf`, read on 2026-10-06. These are requirements for the candidate's submission, not instructions to publish files or send email automatically.

## Scope and precedence

- At least one task must be completed. All three receive priority consideration when performed well.
- Each task is independently scored out of 100; optional task bonuses are described separately.
- Deadline: within two working days of receiving the email. Receipt date/time and the employer's working-day interpretation are not yet known. Do not infer a due date from today's date.
- The email makes a personally narrated video of up to five minutes mandatory. The PDF labels it optional. Use the email's mandatory requirement; do not depend on the PDF's video bonus wording.
- AI assistance is permitted, but architectural choices and code must be understood and defensible by the candidate.
- Use free-tier tooling. Confirm actual current quotas and model availability before selecting external services; PDF examples are not availability guarantees.

## Submission acceptance checklist

| ID | Requirement | Evidence needed |
|---|---|---|
| SUB-01 | Public GitHub repository named `CDAZZDEV-MLE-[YourName]` | Repository opens in a signed-out/incognito browser |
| SUB-02 | Working code and notebooks in `task1_financial/`, `task2_genai/`, `task3_agentic/` for attempted tasks | Reproducible clean-environment run |
| SUB-03 | One Colab notebook per attempted task; outputs retained | Notebook with actual cell results and badge/direct link in task README |
| SUB-04 | Root README, CITATIONS.md, REFLECTION.md | Complete files; reflection no more than 600 words |
| SUB-05 | No committed credentials | Inspect notebook outputs, artifacts, configuration, and Git history before publication |
| SUB-06 | Personal video walkthrough, up to five minutes | Candidate records and narrates; unlisted YouTube or public-view Drive link |
| SUB-07 | Task 2 model accessible if attempted | Saved merged model on HF or Drive; public access or reviewer access where permitted |
| SUB-08 | Task 3 trace if attempted | `task3_agentic/logs/agent_trace.jsonl` committed with real execution records |
| SUB-09 | Large artifact links accessible | Drive set to anyone with the link; test signed out |

The PDF permits private HF reviewer access; the quoted email does not include the reviewer identity. Obtain it only if using that option. GitHub must be public.

## Task coverage

### Task 1: financial research, 100 points

Detailed requirements and acceptance criteria: [task1-financial.md](task1-financial.md).

- Part A, 60: dynamic two-year daily OHLCV via yfinance (10); five indicators without TA-Lib (25); at least ten recent headlines (10); summary dictionary (10); robustness (5).
- Part B, 40: per-headline validated sentiment JSON and aggregation (10); combined-indicator Buy/Hold/Sell reasoning (15); validated outputs and graceful failure logging (10); separated prompts and correct roles (5).
- Optional report bonus, up to 5: one-page Markdown brief rendered to styled HTML or PDF, embedded matplotlib chart, required sections and risk disclaimer.

### Task 2: fine-tuning, 100 points

Do not begin training until these have a task-specific approved spec:

| Group | Required behavior and evidence | Marks |
|---|---|---|
| 2A | Nontrivial domain input/output contract and correctness criteria; at least 100 teacher-generated examples; full generation system prompt; length distributions and topic/keyword frequencies; chat JSONL matching base-model template; 80/10/10 split with sizes | 30 |
| 2B | QLoRA with 4-bit NF4 on Colab free GPU; explain rank, alpha, target modules, learning rate, scheduler, epochs, batch size, accumulation, and max length; actual train/validation loss each epoch; decreasing validation loss; `merge_and_unload()` and accessible saved merged model | 40 |
| 2C | Base and fine-tuned ROUGE-L on same held-out set; BERTScore F1 or rubric-based structured LLM judge; manually label at least 10 fine-tuned responses correct/partial/hallucinated and calculate hallucination rate; two evidence-based analysis paragraphs | 30 |

Teacher and student must be different models. Split by scenario and remove duplicates to avoid leakage. With exactly 100 examples, split sizes are 80/10/10; this meets the minimum but yields a small, uncertain evaluation. Prefer a larger held-out set if resources allow. Document real OOM incidents if encountered. Do not fabricate improvement or logs if validation loss fails to decrease; investigate and report the actual outcome.

Optional bonus: threshold-triggered RAG using ChromaDB and a concrete before/after example. A self-rated confidence is not calibrated probability; document that limitation.

### Task 3: agentic research, 100 points

| Group | Required behavior and evidence | Marks |
|---|---|---|
| 3A | Agent using LangChain, LangGraph, or CrewAI; implements `get_price_data(ticker, period)`, `get_news(ticker, n)`, `calculate_volatility(ticker, window)`, `llm_sentiment(headlines)`, `web_search(query)`; autonomous tool ordering; visible observe/replan; graceful alternative on failure; report with financial health, exactly three supported 90-day risks, and one data-driven hedge | 50 |
| 3B | Analyst has only price, volatility, sentiment tools; Writer has only search and news tools; typed/Pydantic handoff; complete notebook message trace; at least one executed clarification -> response -> incorporation loop; no manual intervention | 35 |
| 3C | Follow-up uses session context without refetching; final brief persisted by ticker/date and reused on subsequent run; every tool call logged with arguments, output truncated to 200 characters, and wall-clock duration | 15 |

Clarify in the Task 3 design how the Analyst receives headlines for its sentiment tool despite having no news tool. Proposed solution: the Writer can pass a typed headline collection to the Analyst; enforce permissions in code. Persist both success and failure calls, and set a bounded tool/step budget to prevent runaway loops. Cache policy should include date and configuration version to avoid silently reusing incompatible reports.

Optional bonus: external observability trace screenshot or Streamlit trace dashboard.

## Evidence integrity

- Mocks and fixtures can verify software behavior but do not count as live market-data, LLM, agent, or training execution evidence.
- Keep actual timestamps, model identifiers, dataset counts, and output artifacts.
- Do not invent metrics, claim a notebook ran when it did not, or write reflection claims before work is completed.
- The candidate must personally perform the video narration and manual response review.
