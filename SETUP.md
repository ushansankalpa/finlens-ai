# Private credentials and Colab setup

## Groq free inference

1. Open [Groq Console](https://console.groq.com/) and sign in/create your account. Stay on the free plan.
2. Open API Keys and create a key. Do not paste it into chat, a notebook cell, README, or committed file.
3. For Colab, open the key icon in the left sidebar, add a secret named `GROQ_API_KEY`, and enable notebook access.
4. Select a model your account permits. Default inference model is `openai/gpt-oss-20b`; proposed teacher is `openai/gpt-oss-120b`. Both remain configurable. Check the [current model list](https://console.groq.com/docs/models) and [account limits](https://console.groq.com/docs/rate-limits).
5. Run the notebook's secret-loading and smoke-check cells. Authentication or rate-limit errors must be resolved before a full run.

Local PowerShell alternative, without echoing the key:

```powershell
$assessmentKey = Read-Host 'Groq API key' -AsSecureString
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new('', $assessmentKey).Password
$env:GROQ_MODEL = 'openai/gpt-oss-20b'
.\.venv\Scripts\python.exe -m task1_financial.pipeline --ticker AAPL
```

The key lives in that shell process and its children. `.env.example` is only a template; the implementation does not automatically load `.env` files.

## Colab GPU for Task 2

1. Open [Google Colab](https://colab.research.google.com/) and upload `task2_genai/assessment.ipynb` once available.
2. Choose Runtime -> Change runtime type -> a free GPU such as T4 if offered. Free GPU access is not guaranteed.
3. Run the installation and hardware-check cells. They verify CUDA before model download/training.
4. Mount Drive when prompted so generated datasets, training logs, and adapters survive a runtime reset. Keep keys in Colab Secrets, never Drive code/config files.
5. Run the NF4 smoke test before the complete training run. An unavailable GPU requires waiting for free capacity or revising the task scope; do not present CPU-only scaffolding as QLoRA execution.

## Hugging Face model upload

1. Create/sign in at [Hugging Face](https://huggingface.co/).
2. Create an appropriately scoped write token and save it in Colab Secrets as `HF_TOKEN`.
3. Set your final model repository ID in the notebook. Upload the merged model and tokenizer after successful reload/inference verification.
4. Check the public model page signed out. If using private access, obtain the reviewer identity from the employer first.

## Local install and checks

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest
```

Task 2's heavy training dependencies are installed separately in its Colab notebook; they are not needed for local Task 1/3 checks.
