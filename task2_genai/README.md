# Task 2: supply-chain exception explanation

Spec: [task2-genai.md](../specs/task2-genai.md). Notebook: [assessment.ipynb](assessment.ipynb).

Status: implementation and notebook are present; real Qwen tokenizer/response masking verified in `outputs/tokenizer_check.json`. Teacher generation, GPU training, merge/reload, evaluation, personal review, and publication have not run.

The domain contract maps observed facts to six exception categories under supplied thresholds and explicit precedence. Teacher labels, severity, actions, and evidence are validated against a deterministic policy oracle. Industry groups are disjoint across 80/10/10 splits. Cross-split near-duplicate request flags stop training for inspection.

Use the notebook on a Colab CUDA GPU. It runs NF4 feasibility, resumable teacher generation, diversity analysis, assistant-only QLoRA, real per-epoch losses, validation-selected checkpoint export, non-quantized merge, matched base/merged inference, ROUGE-L, BERTScore, and task-specific checks. Heavy dependencies remain unverified until the Colab smoke run.

Candidate actions: privately configure Groq/HF tokens, review dataset quality, label at least ten generated responses personally, write two evidence-based paragraphs, fill the model card, then enable upload. Do not infer success from the presence of scripts.

Drive persistence is the notebook default. Before GitHub submission, copy generated JSONL, diversity report, exact teacher prompt, training configuration/hardware/loss logs, evaluation outputs, and personal-review artifacts into the task folder. Keep merged weights on HF/Drive. Include real model and Colab links here after execution/publication.
