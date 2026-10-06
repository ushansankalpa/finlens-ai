"""Create editable Colab notebooks. This script never fabricates cell outputs."""
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nb.v4.new_markdown_cell(text)


def code(text):
    return nb.v4.new_code_cell(text)


BOOTSTRAP = '''from pathlib import Path
import os, sys, subprocess, zipfile

if not Path("pyproject.toml").exists():
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("Run locally from repository root or upload assessment_code.zip in Colab")
    print("Upload assessment_code.zip from the project's dist folder.")
    uploads = files.upload()
    bundle = next((name for name in uploads if name.endswith(".zip")), None)
    if bundle is None:
        raise RuntimeError("Code bundle ZIP required")
    target = Path("/content/cdazzdev_assessment").resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise ValueError("Unsafe ZIP member")
        archive.extractall(target)
    os.chdir(target)
from importlib.metadata import version, PackageNotFoundError
try:
    version("cdazzdev-mle-assessment")
except PackageNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"])
print("Workspace:", Path.cwd())'''

SECRETS = '''import os
try:
    from google.colab import userdata
    for name in ["GROQ_API_KEY", "HF_TOKEN"]:
        try:
            value = userdata.get(name)
            if value:
                os.environ[name] = value
        except Exception:
            print(name, "not configured or notebook access disabled")
except ImportError:
    print("Local run uses environment variables.")
from shared.preflight import check
print(check(inference=bool(os.getenv("GROQ_API_KEY"))))'''


def save(task, cells):
    path = ROOT/task/"assessment.ipynb"
    if path.exists() and any(cell.get("outputs") for cell in nb.read(path,as_version=4).cells):
        raise RuntimeError(f"Refusing to overwrite executed outputs in {path}; edit the existing notebook")
    notebook = nb.v4.new_notebook(cells=cells,metadata={
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python","version":"3.11"},
        "colab":{"name":f"{task}_assessment.ipynb"}})
    nb.validate(notebook)
    nb.write(notebook,path)


def main():
    save("task1_financial",[
        md("# Task 1: LLM-powered equity research\n\nSee `specs/task1-financial.md`. This notebook separates actual data evidence from LLM execution. Retain outputs after running. Read `SETUP.md` before supplying secrets."),
        code(BOOTSTRAP),code(SECRETS),
        code('''TICKER = "AAPL"
RUN_LLM = bool(os.getenv("GROQ_API_KEY"))
REUSE_SAVED_DATA = True  # Saved data is labeled with its real retrieval timestamp; set False to refresh.
print("LLM enabled:", RUN_LLM)'''),
        md("## Numerical and robustness checks\nThese use independently calculated expectations and clearly labeled fixtures; they are not live LLM evidence."),
        code('''import subprocess, sys
subprocess.run([sys.executable, "-m", "pytest", "tests/test_financial.py", "tests/test_llm.py"], check=True)'''),
        md("## Real market data and recent news\nRead the retrieval timestamp below. Reuse is explicit; no stale result is presented as a newly fetched quote."),
        code('''import json
from pathlib import Path
from task1_financial.data_only import run as data_run
saved = Path(f"task1_financial/outputs/{TICKER}_data_evidence.json")
if REUSE_SAVED_DATA and saved.exists():
    evidence = json.loads(saved.read_text(encoding="utf-8"))
    print("Using saved real data; retrieved_at:", evidence["retrieved_at"])
else:
    evidence = data_run(TICKER)
print(json.dumps(evidence, indent=2))
assert len(evidence["headlines"]) >= 10, "News requirement not satisfied"
assert evidence["row_count"] > 400, "Inspect actual history coverage"'''),
        md("## Structured sentiment and indicator-combination reasoning\nA missing key results in an explicit incomplete stage, not fabricated sentiment."),
        code('''from task1_financial.pipeline import run
if RUN_LLM:
    result, frame = run(TICKER)
    print(result.model_dump_json(indent=2))
    assert result.signal is not None, "Signal inference incomplete"
    assert len(result.sentiments) == len(result.headlines), "Some sentiment calls failed"
else:
    print("NOT EXECUTED: configure GROQ_API_KEY and rerun for Task 1B evidence.")'''),
        md("## Research brief\nWhen inference is unavailable, the preview explicitly says the recommendation is unavailable. It is not a complete Task 1 submission."),
        code('''import pandas as pd
from IPython.display import HTML, display
from task1_financial.schemas import ResearchResult, MarketSummary
from task1_financial.report import render_report
if not RUN_LLM:
    frame = pd.read_csv(f"task1_financial/outputs/{TICKER}_indicators.csv", index_col="Date", parse_dates=True)
    result = ResearchResult(summary=MarketSummary.model_validate(evidence["summary"]),
        indicators=evidence["indicators"],headlines=evidence["headlines"],sentiments={},aggregate_sentiment=None,
        signal=None,warnings=evidence["warnings"] + ["LLM not executed: data-only preview"],
        metadata={"stage":"data_only_preview","retrieved_at":evidence["retrieved_at"]})
report_path = render_report(result, frame, "task1_financial/outputs")
display(HTML(report_path.read_text(encoding="utf-8")))
print("Report:", report_path)'''),
        md("## Candidate review\nExplain Wilder seed versus generic exponential smoothing, EMA warmup, population Bollinger deviation, adjusted-price policy, sentiment aggregation limitations, and whether actual signal reasoning combines indicators. Finish citations/reflection using actual outcomes."),
    ])
    save("task2_genai",[
        md("# Task 2: supply-chain QLoRA and matched evaluation\n\nSee `specs/task2-genai.md` for the input/output contract and all hyperparameter justifications. This notebook is unexecuted until run on a CUDA Colab GPU. Do not claim training/evaluation before outputs exist."),
        code(BOOTSTRAP),code(SECRETS),
        code('''import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", ".[training]"])
import torch
assert torch.cuda.is_available(), "Select Runtime -> Change runtime type -> GPU"
print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__)
from importlib.metadata import version
print({name:version(name) for name in ["transformers","peft","bitsandbytes","accelerate"]})'''),
        code('''# Persist progress outside the ephemeral runtime.
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACT_ROOT = Path("/content/drive/MyDrive/CDAZZDEV_MLE/task2")
except ImportError:
    ARTIFACT_ROOT = Path("task2_genai")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
DATA_DIR = ARTIFACT_ROOT / "data"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
MERGED_DIR = ARTIFACT_ROOT / "models/merged"
OUTPUT_DIR = ARTIFACT_ROOT / "outputs"
MODEL_REPO_ID = ""  # Set your HF username/model-name only when ready to publish.
RUN_TRAINING = True
RUN_EVALUATION = True
PUBLISH_MODEL = False'''),
        md("## NF4 feasibility gate\nThis is one forward/backward test; it is not full fine-tuning evidence."),
        code('''from task2_genai.train import TrainingConfig, RATIONALE, smoke_test
from dataclasses import asdict
config = TrainingConfig()
print(asdict(config))
print(RATIONALE)
print(smoke_test(config))'''),
        md("## Teacher dataset and explicit oracle\nTeacher and student differ. Generation is resumable, validates labels/evidence against supplied policy, and saves the exact teacher system prompt."),
        code('''from task2_genai.dataset import generate
from task2_genai.domain import TEACHER_SYSTEM, STUDENT_SYSTEM
print("Teacher system prompt:\\n", TEACHER_SYSTEM)
print("Student system prompt:\\n", STUDENT_SYSTEM)
diversity = generate(DATA_DIR, count=200)
print(diversity)
assert not diversity["near_duplicate_cross_split_flags"], "Resolve near duplicates before training"'''),
        code('''import matplotlib.pyplot as plt
import pandas as pd
fig, axes = plt.subplots(1, 2, figsize=(11, 3))
for split, info in diversity["splits"].items():
    axes[0].hist(info["prompt_char_lengths"], alpha=.5, label=split)
axes[0].set_title("Prompt character lengths"); axes[0].legend()
pd.DataFrame({split:info["topic_counts"] for split,info in diversity["splits"].items()}).fillna(0).plot.bar(ax=axes[1])
axes[1].set_title("Topic frequencies"); plt.tight_layout(); plt.show()
print("Keyword frequencies:", diversity["keyword_counts"])
print("Split sizes:", {split:info["count"] for split,info in diversity["splits"].items()})'''),
        md("## Actual QLoRA training\nBase weights are frozen NF4, adapters receive assistant-only loss, validation controls checkpoint selection. Inspect all real losses; do not fabricate a decreasing curve."),
        code('''from task2_genai.train import train
if RUN_TRAINING:
    history = train(DATA_DIR, CHECKPOINT_DIR, config)
    display(pd.DataFrame(history))
    pd.DataFrame(history).plot(x="epoch", y=["train_loss","validation_loss"], marker="o")
    plt.show()
else:
    print("Training skipped; execution evidence incomplete")'''),
        md("## Merge and reload\nThe non-quantized base is reloaded for `merge_and_unload()`. Evaluation below reloads the saved merged weights and runs inference."),
        code('''from task2_genai.train import merge
if RUN_TRAINING:
    print("Saved merged model:", merge(CHECKPOINT_DIR / "best_adapter", MERGED_DIR, config))'''),
        md("## Frozen test-set baseline comparison\nBoth models receive the same system prompt, chat inputs, deterministic generation settings, and held-out examples. ROUGE-L and BERTScore are supplemented by schema/category/severity checks."),
        code('''from task2_genai.evaluate import evaluate
if RUN_EVALUATION:
    comparison = evaluate(DATA_DIR, MERGED_DIR, OUTPUT_DIR, config.model_id)
    display(pd.DataFrame(comparison).T)
    print((OUTPUT_DIR / "comparison.md").read_text())
else:
    print("Evaluation skipped; execution evidence incomplete")'''),
        md("## Personal response review\nOpen `manual_review.csv` in Drive. Personally label at least ten rows `correct`, `partially_correct`, or `hallucinated`, and write reasons. Then rerun the next cell. An unfilled review does not count. Write the two analysis paragraphs in `qualitative_analysis.md`."),
        code('''from task2_genai.evaluate import summarize_review
try:
    print(summarize_review(OUTPUT_DIR / "manual_review.csv"))
except ValueError as error:
    print("PENDING PERSONAL REVIEW:", error)'''),
        md("## Model card and optional publication step\nSet the repository ID and enable publication only after inspecting the model and results. No credentials appear in the notebook."),
        code('''from task2_genai.evaluate import publish
import shutil
if MERGED_DIR.exists() and not (MERGED_DIR / "README.md").exists():
    shutil.copyfile("task2_genai/MODEL_CARD_TEMPLATE.md", MERGED_DIR / "README.md")
if PUBLISH_MODEL:
    assert MODEL_REPO_ID and "/" in MODEL_REPO_ID, "Set your actual HF repository ID"
    print(publish(MERGED_DIR, MODEL_REPO_ID))
else:
    print("Not published. Fill the model card with actual results before uploading.")'''),
        md("## Download submission evidence\nCopy data/outputs and relevant training logs from Drive into the GitHub task folder. Keep merged weights on HF/Drive. Download this executed notebook with its outputs. Final status requires real results and the personal review."),
    ])
    save("task3_agentic",[
        md("# Task 3: autonomous financial agents\n\nSee `specs/task3-agentic.md`. Tool order is selected by the model from observations; Analyst and Writer have enforced separate registries. Tests use fixtures; live traces below are separate."),
        code(BOOTSTRAP),code(SECRETS),
        code('''TICKER = "AAPL"
RUN_LIVE = bool(os.getenv("GROQ_API_KEY"))
print("Live agent enabled:", RUN_LIVE)'''),
        md("## Control-flow checks\nTests prove permissions, observed failure -> fallback, typed clarification, memory reuse, cache, and JSONL format. Temporary fixture logs are not submitted as live traces."),
        code('''subprocess.run([sys.executable, "-m", "pytest", "tests/test_agents.py"], check=True)'''),
        md("## Single-agent autonomous research"),
        code('''from task3_agentic.engine import ResearchSystem
import json
if RUN_LIVE:
    single = ResearchSystem()
    single_result = single.run(TICKER, mode="single", refresh=True)
    for message in single.messages:
        print(json.dumps(message, indent=2))
    print(json.dumps(single_result["report"], indent=2))
else:
    print("NOT EXECUTED: configure GROQ_API_KEY for live tool/observe/replan evidence")'''),
        md("## Restricted two-agent pipeline and critique loop"),
        code('''if RUN_LIVE:
    multi = ResearchSystem()
    multi_result = multi.run(TICKER, mode="multi", refresh=True)
    for message in multi.messages:
        print(json.dumps(message, indent=2))
    print(json.dumps(multi_result["report"], indent=2))
    assert multi.clarification_count == 1, "Required critique loop did not execute"'''),
        md("## Short-term memory: no tool refetch"),
        code('''if RUN_LIVE:
    before = multi.tool_calls
    memory_answer = multi.follow_up("What quantitative risk measure did you already retrieve? Use only existing evidence.")
    print(json.dumps(memory_answer, indent=2))
    assert multi.tool_calls == before'''),
        md("## Persistent cache: second session uses saved report"),
        code('''if RUN_LIVE:
    cached = ResearchSystem()
    cached_result = cached.run(TICKER, mode="multi")
    print("Cache hit:", cached_result["cache_hit"], "New tool calls:", cached_result["new_tool_calls"])
    assert cached_result["cache_hit"] and cached_result["new_tool_calls"] == 0'''),
        md("## Controlled live failure recovery\nThis separate run injects a single search failure, then exposes real tools. Inspect the trace for an alternative decision. It is explicitly labeled failure injection, not an organic source outage."),
        code('''from task3_agentic.tools import FinancialTools
from shared.llm import JSONClient
class OneSearchFailure(FinancialTools):
    def __init__(self, client):
        super().__init__(client)
        self.failed = False
    def web_search(self, query):
        if not self.failed:
            self.failed = True
            raise RuntimeError("Controlled one-time search failure")
        return super().web_search(query)
if RUN_LIVE:
    client = JSONClient()
    failure_tools = OneSearchFailure(client)
    failure_system = ResearchSystem(client, failure_tools, log_dir="task3_agentic/logs/failure_demo",
                                    cache_dir="task3_agentic/cache/failure_demo")
    failure_result = failure_system.run(TICKER, mode="single", refresh=True)
    for message in failure_system.messages:
        print(json.dumps(message, indent=2))
    print("Injected failure executed:", failure_tools.failed)
    assert failure_tools.failed, "Agent did not select web_search; this run does not prove injected recovery"'''),
        md("## Actual observability artifact"),
        code('''from pathlib import Path
path = Path("task3_agentic/logs/agent_trace.jsonl")
if path.exists():
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for entry in entries:
        print(json.dumps(entry))
    assert all(len(entry["output"]) <= 200 for entry in entries)
else:
    print("NOT GENERATED: no live tool calls have run; Task 3 evidence incomplete")'''),
        md("## Candidate interpretation\nExplain actual tool choices, failure alternatives, enforced restrictions, typed headlines passed to Analyst, clarification incorporation, cache freshness, and the distinction between search snippets and verified source facts. Download the notebook and actual logs with outputs retained."),
    ])


if __name__ == "__main__":
    main()
