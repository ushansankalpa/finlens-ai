# CDAZZDEV - ML Technical

Spec-driven implementation of all three assessment tasks. Start with [SETUP.md](SETUP.md), then the relevant task notebook. **This is not yet a complete submission:** private inference credentials, actual Colab training/evaluation, live agent traces, personal review, and video are still required.

## Current state

| Task | Implementation | Actual execution |
|---|---|---|
| [Financial AI](task1_financial/README.md) | Data/indicators/news, validated inference, report renderer | 751 real AAPL daily bars and ten recent headlines; locally executed data-only notebook; LLM pending |
| [Generative AI](task2_genai/README.md) | Teacher/oracle validation, grouped split, NF4 QLoRA, merge, baseline metrics, review template | GPU training/evaluation and teacher generation pending |
| [Agentic workflows](task3_agentic/README.md) | LangGraph decisions, five tools, restricted roles, typed critique, memory, trace | Fixture verification and local notebook execution; live stages explicitly skipped without credentials |

See [EVIDENCE.md](EVIDENCE.md) for acceptance status and limitations. Executed notebook cell counts alone do not establish that skipped live stages ran.

## Local quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m shared.preflight
.\.venv\Scripts\python.exe -m task1_financial.data_only --ticker AAPL
```

After privately configuring GROQ_API_KEY as described in SETUP.md:

```powershell
.\.venv\Scripts\python.exe -m task1_financial.pipeline --ticker AAPL
.\.venv\Scripts\python.exe -m task3_agentic.engine --ticker AAPL --mode multi --refresh
```

Use `task2_genai/assessment.ipynb` on a free Colab CUDA GPU for training. The student is Qwen2.5-1.5B-Instruct; the proposed configurable teacher is GPT-OSS-120B on Groq. Actual account quotas and GPU capacity must pass a smoke check. Models and code are downloaded from their providers; API keys/tokens remain private.

## Colab without an existing GitHub repository

`dist/assessment_code.zip` is a generated portable bundle. Upload the selected `task*/assessment.ipynb` to Colab, then upload that bundle when its first cell asks. Read SETUP.md for Secrets/GPU/Drive instructions. The bundle contains code, specs, tests, notebooks, and saved data-only evidence, with environments/credentials/model weights excluded.

Rebuild after changes:

```powershell
.\.venv\Scripts\python.exe scripts/build_bundle.py
```

`scripts/build_notebooks.py` refuses to overwrite notebooks containing executed outputs. Edit existing notebooks directly after execution. `scripts/execute_local_notebook.py` preserves real outputs, including explicit pending stages, using the workspace interpreter.

## Architecture and spec-driven workflow

- `shared/`: bounded inference transport, Pydantic validation, strict JSON serialization, preflight.
- `task1_financial/`: deterministic math and sourced data separated from prompt/inference logic.
- `task2_genai/`: teacher records checked against policy; industry-group holdout; assistant-only loss; validation-selected checkpoint; matched evaluation.
- `task3_agentic/`: model-selected actions inside LangGraph; role-specific permissions; typed evidence copied from observations; critique checkpoint; session/persistent memory.
- `tests/`: numerical, invalid-data, transport, agent-control, policy, and masking checks. Fixtures are never submitted as live traces.

Follow [PLAN.md](PLAN.md) and [assessment requirements](specs/assessment-requirements.md). Implement a requirement group, verify it, save evidence, update the ledger/citations, then proceed.

## Reproducibility and limitations

`pyproject.toml` defines compatible dependency ranges; `requirements-local.lock.txt` records the tested local environment. The heavy GPU stack is not yet verified. Colab notebooks print actual versions and hardware. External data/API outputs change over time. Prices are adjusted completed daily bars, LLM confidence is heuristic, and search snippets do not establish facts beyond their contents. Synthetic policy performance cannot establish real supply-chain reliability.

No trading execution is implemented. Financial output is an assessment research demonstration, not investment advice.

## Submission remaining

Create public `CDAZZDEV-MLE-[YourName]`, add real Colab links, retain all required outputs, upload/verify Task 2 merged weights, commit actual Task 3 traces, finish [CITATIONS.md](CITATIONS.md) and personal [REFLECTION.md](REFLECTION.md) (<=600 words), and personally record/narrate a <=5-minute video. The email makes video mandatory. Verify links signed out. The confidential PDF is excluded.
