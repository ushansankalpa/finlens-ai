# Task 1: equity research

Spec: [task1-financial.md](../specs/task1-financial.md). Notebook: [assessment.ipynb](assessment.ipynb).

Current execution status: real AAPL data/indicator/news evidence exists in `outputs/AAPL_data_evidence.json`; LLM inference awaits a private Groq credential. Data-only execution is not complete Task 1 evidence.

```powershell
.\.venv\Scripts\python.exe -m task1_financial.data_only --ticker AAPL
.\.venv\Scripts\python.exe -m task1_financial.pipeline --ticker AAPL
```

Run from repository root after following [SETUP.md](../SETUP.md). Pipeline output is JSON, indicator CSV, Markdown, styled HTML, and a chart. Exit code 2 indicates a partial inference/news result; inspect warnings. Data fetch failure exits with a clear error.

Indicators use adjusted OHLC, first-observation recursive EMA, explicit Wilder RSI initialization, and population Bollinger standard deviation. Missing P/E is null. The current-price field is labeled latest completed adjusted close. Summary and report are research demonstrations with a risk disclaimer.

Colab: upload this notebook and `dist/assessment_code.zip`. The bootstrap cell extracts the bundle; select `REUSE_SAVED_DATA=False` for fresh data. Add `GROQ_API_KEY` in Colab Secrets and rerun all cells for complete inference evidence. Save the notebook with outputs.

After GitHub publication, add the real Colab direct link/badge here. No repository URL has been created yet.
