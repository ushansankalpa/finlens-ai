# Task 1: equity research

Spec: [task1-financial.md](../specs/task1-financial.md). Notebook: [assessment.ipynb](assessment.ipynb).

Current execution status: live AAPL pipeline and local notebook completed on 2026-10-07 (Asia/Colombo). `outputs/AAPL_research.json` contains ten validated live headline sentiments and a validated three-sentence recommendation grounded in computed indicators, with no warnings. The executed notebook retains outputs. Colab execution/link and final candidate review remain submission steps.

```powershell
.\.venv\Scripts\python.exe -m task1_financial.data_only --ticker AAPL
.\.venv\Scripts\python.exe -m task1_financial.pipeline --ticker AAPL
```

Run from repository root after following [SETUP.md](../SETUP.md). Pipeline output is JSON, indicator CSV, Markdown, styled HTML, and a chart. Exit code 2 indicates a partial inference/news result; inspect warnings. Data fetch failure exits with a clear error.

Indicators use adjusted OHLC, first-observation recursive EMA, explicit Wilder RSI initialization, and population Bollinger standard deviation. Missing P/E is null. The current-price field is labeled latest completed adjusted close. Summary and report are research demonstrations with a risk disclaimer.

Colab: upload this notebook and `dist/assessment_code.zip`. The bootstrap cell extracts the bundle; select `REUSE_SAVED_DATA=False` for fresh data. Add `GROQ_API_KEY` in Colab Secrets and rerun all cells for complete inference evidence. Save the notebook with outputs.

After GitHub publication, add the real Colab direct link/badge here. No repository URL has been created yet.
