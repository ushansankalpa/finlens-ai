# Task 3: financial research agents

Spec: [task3-agentic.md](../specs/task3-agentic.md). Notebook: [assessment.ipynb](assessment.ipynb).

Status: implementation and fixture-based control-flow verification exist; live LLM-driven agents and assessment trace remain pending a private Groq credential.

```powershell
.\.venv\Scripts\python.exe -m task3_agentic.engine --ticker AAPL --mode single --refresh
.\.venv\Scripts\python.exe -m task3_agentic.engine --ticker AAPL --mode multi --refresh
```

The LangGraph loop alternates validated model decisions and actual observations. Tool order is not prescribed. Analyst can use only prices/volatility/sentiment; Writer can use only search/news. Typed DataBrief copies source observations using validated evidence IDs. Typed headlines can cross to Analyst for sentiment. Writer must execute one specific clarification before report finalization.

Every dispatched tool call, including rejected/failed calls, records inputs, <=200-character output, status, duration, agent and run ID in `logs/agent_trace.jsonl`. Complete messages are saved separately and displayed in the notebook. No fixture trace is committed as live evidence.

Follow-up answers have no tool access and must cite session evidence. Cache files use ticker/date plus model/mode/schema/step configuration hash. A compatible second session loads them without fetching. Use `--refresh` for fresh data. Errors/budget exhaustion leave trace evidence and an explicit incomplete run.

The notebook separately demonstrates single-agent research, two-agent critique, memory, cache, and a labeled one-time live search failure. If the model never selects search, that injected run does not establish recovery and its assertion fails visibly.

Follow [SETUP.md](../SETUP.md), upload notebook/code bundle to Colab, configure secrets, run, and retain outputs. Add actual Colab link and commit actual trace after live execution.
