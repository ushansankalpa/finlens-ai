# Task 2 specification

Status: initial implementation present for the user's all-three scope. Assessment requirements: PDF pages 7-9. No teacher dataset has been generated and no training has run; see ../EVIDENCE.md. Training choices remain subject to the actual GPU smoke test.

## Use case and correctness contract

Domain: supply-chain exception explanation under a supplied policy. Input: a short inventory/shipment record, relevant explicit policy thresholds, and observed facts. Output: structured JSON with `exception_type`, `severity`, `evidence`, `recommended_action`, and a concise `explanation`.

Exception families: stockout, excess inventory, supplier delay, demand surge, quality hold, and insufficient information. This is contextual classification/explanation, not a generic chatbot. Correctness requires the right category/severity under the supplied policy, supported evidence, and a permitted action. Unsupported facts, invented supplier causes, or invented quantities are incorrect. Policy conflicts or missing essential fields should produce insufficient-information behavior.

Proposed student: `Qwen/Qwen2.5-1.5B-Instruct`, a small instruct model with an official model card. Its size is a resource-oriented choice, not evidence of GPU compatibility or task accuracy. Verify the exact installed stack with a smoke run before generating the full dataset. Choose a different capable teacher available on the current free inference account and record its exact ID.

## Dataset design

- Generate 200 valid teacher examples, targeting 160 train / 20 validation / 20 test. Minimum required by assessment is 100; this target allows 20 held-out responses while retaining 80/10/10.
- Cover all six exception families, multiple industries, short/long contexts, numerical boundary cases, missing fields, normal/contradictory evidence, and irrelevant distractors. Report actual distributions.
- Assign scenario-family IDs before splitting; keep near-duplicate scenarios in one split and deduplicate before freezing the split. Adjust group construction to maintain exact target counts. Save seed and hashes. Do not inspect test outcomes while choosing checkpoints.
- Validate schema, supplied-policy consistency, evidence provenance, and duplicates. Teacher outputs are fallible; spot-check across every family.
- Store JSONL records with `messages` containing system/user/assistant turns. Render for training using the student's tokenizer chat template rather than manually inventing role delimiters. Keep scenario metadata outside rendered chat text.
- Analyze token/character lengths, topic/keyword frequencies, class balance, duplicate counts, and truncation counts. Max sequence length must preserve complete target JSON.

### Draft teacher system prompt

Save the exact final prompt used, even if it changes from this draft:

> You generate synthetic training examples for supply-chain exception explanation. For each requested scenario, invent a self-contained record and an explicit policy that make the expected classification verifiable from the provided input alone. Cover stockout, excess inventory, supplier delay, demand surge, quality hold, and insufficient information. Vary industries, context lengths, thresholds, distractors, missing fields, and boundary cases; avoid near duplicates. Return a JSON object with scenario_id, topic, user_input, and assistant_output. assistant_output must have exception_type, severity, evidence, recommended_action, and explanation. Evidence must quote or reference supplied facts; do not invent facts in the answer. Actions must follow the supplied policy. Include insufficient-information examples when essential facts are absent or policies conflict. Do not introduce external knowledge or instructions outside the requested schema.

## Initial training configuration and rationale

These are starting choices to validate against memory and dataset lengths, not measured optimal settings:

| Parameter | Proposed value | Reason |
|---|---|---|
| Quantization | 4-bit NF4, double quantization | Meets QLoRA requirement and reduces base-weight memory |
| Compute dtype | FP16 on T4; supported BF16 only after capability check | Match actual hardware capability |
| LoRA rank | 16 | Moderate adapter capacity for a small domain dataset |
| LoRA alpha | 32 | Explicit adapter scale of alpha/r = 2 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Adapt attention and feed-forward transforms; assert names exist |
| LoRA dropout | 0.05 | Regularize small-dataset fitting |
| Bias/task type | none / causal LM | Train adapters for autoregressive output without bias adaptation |
| Learning rate | 0.0002 | Starting adapter-learning rate; inspect held-out loss for instability |
| Scheduler/warmup | cosine / ratio 0.1 | Smooth learning-rate decay and gradual warmup |
| Epochs | 3 | Multiple per-epoch loss observations; select by validation only |
| Microbatch/accumulation | 1 / 8 | Effective batch 8 on one GPU with low activation memory |
| Max sequence length | 1024 initially | Bounded memory; revise based on measured lengths to avoid corrupting targets |
| Optimizer/weight decay | AdamW / 0.01 | Standard stable adapter optimization with explicit regularization |
| Gradient clipping/checkpointing | max norm 1.0 / enabled | Limit gradient spikes and activation memory |
| Seed | 42 | Repeatable split and training configuration |
| Loss masking | Assistant response tokens only | Train requested response behavior without scoring the prompt; verify masks |

Record and explain all additional training configuration choices and relevant defaults in the notebook. Loss masking implementation and library compatibility require a small test. Log hardware, package versions, seed, configuration, actual time and memory. Evaluation uses the same masking convention each epoch.

## Acceptance and evidence

| ID | Requirement | Gate |
|---|---|---|
| T2-01 | Domain contract and success/failure examples | Demonstrate one correct, incorrect, and insufficient-information example |
| T2-02 | At least 100 teacher examples, teacher differs from student, full prompt saved | Actual IDs, generation metadata and validated dataset counts |
| T2-03 | Diversity and uncontaminated 80/10/10 chat JSONL split | Distribution plots/tables, dedup/group checks, tokenizer-rendered example |
| T2-04 | Actual NF4 QLoRA on free Colab GPU | Quantization config, adapter parameters, successful training output |
| T2-05 | Hyperparameters explained; train/validation losses per epoch | Real loss table/plot, decreasing validation-loss evidence or honestly documented failure |
| T2-06 | Merged model saved and accessible | Reload unquantized compatible base, attach adapter, `merge_and_unload()`, save weights/tokenizer, reload and infer, verify shared link |
| T2-07 | Base vs fine-tuned ROUGE-L on identical frozen test set | Same system prompt/input/template/generation limits; complete outputs and comparison table |
| T2-08 | Additional metric | BERTScore F1 on same response field for both models, with scorer/version recorded |
| T2-09 | At least ten personally reviewed responses | Candidate labels correct/partially correct/hallucinated with reasons; rate = hallucinated/reviewed * 100 |
| T2-10 | Two paragraphs explaining results and failure modes | Specific output IDs, observed examples, and next steps |
| T2-11 | Executed Colab notebook, model card and model link | Outputs retained; reproducible instructions; synthetic-data and intended-use limitations |

Compute required ROUGE-L consistently over canonical rendered complete responses; additionally report JSON-valid rate and exact category/severity accuracy because text overlap alone cannot establish policy correctness. BERTScore does not replace grounded/manual checks. Freeze checkpoint selection and generation policy before test evaluation. Quantify how small test-set size limits conclusions.

Save adapters before releasing GPU memory for a non-quantized base reload and merge; budget CPU RAM and disk. Adapter-only upload does not satisfy the merged-model requirement. If merge or improvement fails, report it and repair where possible; do not relabel an unexecuted or unsuccessful result as completed.

## References consulted for the proposed design

- [Official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
- [PEFT LoRA and merge API](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md).
- [Groq rate limits](https://console.groq.com/docs/rate-limits); account/model limits must be checked at execution time.

These are planning references, not claims that their source code was adapted.
