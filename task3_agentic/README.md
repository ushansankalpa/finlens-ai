# Task 3: financial research agents

Spec: [task3-agentic.md](../specs/task3-agentic.md). Notebook: [assessment.ipynb](assessment.ipynb).

Status: implementation and fixture checks pass. A preliminary live multi-agent report and actual trace exist, but news coverage and grounding remain incomplete. The next live attempt stopped on Groq HTTP 429. Single-agent, live memory/cache, failure-recovery and full notebook evidence remain pending. See logs/RUN_AUDIT.md; do not present preliminary results as a completed task.

```powershell
.\.venv\Scripts\python.exe -m task3_agentic.engine --ticker AAPL --mode single --refresh --model openai/gpt-oss-120b
.\.venv\Scripts\python.exe -m task3_agentic.engine --ticker AAPL --mode multi --refresh --model openai/gpt-oss-120b
```

The LangGraph loop alternates validated model decisions and actual observations. Tool order is not prescribed. Analyst can use only prices/volatility/sentiment; Writer can use only search/news. Typed DataBrief copies source observations using validated evidence IDs. Typed headlines can cross to Analyst for sentiment. Writer must execute one specific clarification before report finalization.

Every dispatched tool call, including rejected/failed calls, records inputs, <=200-character output, status, duration, agent and run ID in `logs/agent_trace.jsonl`. Complete messages are saved separately and displayed in the notebook. No fixture trace is committed as live evidence.

Follow-up answers have no tool access and must cite session evidence. Cache files use ticker/date plus model/mode/schema/step configuration hash. A compatible second session loads them without fetching. Use `--refresh` for fresh data. Errors/budget exhaustion leave trace evidence and an explicit incomplete run.

The notebook separately demonstrates single-agent research, two-agent critique, memory, cache, and a labeled one-time live search failure. If the model never selects search, that injected run does not establish recovery and its assertion fails visibly.

Follow [SETUP.md](../SETUP.md), upload notebook/code bundle to Colab, configure secrets, run, and retain outputs. Add actual Colab link and commit actual trace after live execution.
