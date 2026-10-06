# Implementation and submission plan

Status: all three tasks requested, with a 24-hour planning budget. Exact receipt date and deadline remain unknown. Estimates below are time budgets, not promises; full completion depends on actual GPU/API access and execution results.

## Revised 24-hour budget for all three

| Work window | Focus | Gate |
|---|---|---|
| Hours 0-1 | Review all specs, check Colab GPU, HF access and free inference; lock Task 2 use case | Tiny NF4 forward/training step works; provider answers one validated request |
| Hours 1-5 | Task 1 data, indicators, news and validation | Correct indicators and at least ten live headlines |
| Hours 5-8 | Task 2 teacher data, diversity, split, baseline and QLoRA launch | Dataset validated, held-out set frozen, actual training running |
| Hours 8-11 | Finish Task 1 LLM/report/notebook; monitor training between work packets | Task 1 complete; genuine train/validation logs saved |
| Hours 11-16 | Task 3 autonomous agent, two roles, clarification loop, memory and trace | Full live run, critique, memory/cache demonstrations |
| Hours 16-20 | Task 2 merge/reload, matched evaluation, personal ten-response review, upload | Real metric table, saved accessible merged model |
| Hours 20-24 | Clean notebook reruns, docs, citations, reflection, video, access checks | Submission checklist complete |

Start GPU feasibility early. Save dataset and checkpoints outside the ephemeral Colab runtime. Overlapping training with other work assumes the GPU runtime stays connected; it is not guaranteed. If a gate fails, revise the scope openly rather than claiming an incomplete task is complete. Protect the last four hours for evidence and delivery. Skip optional RAG and observability dashboard bonuses initially.

The original Task 1-first breakdown below remains a reference for individual work packets; the revised table controls the all-three schedule.

## Day 1: establish a complete Task 1 path

1. Scope/spec review, 30-45 minutes. Confirm task choice and ticker; lock indicator conventions, output contracts, and definition of done. Create a short decision log when alternatives matter.
2. Data and indicators, 2-3 hours. Implement T1-01 through T1-08. Verify formulas using independently calculated fixtures, then run live data/news retrieval. Save provenance. If fewer than ten headlines are available, implement an actual alternative free source.
3. Validated LLM path, 2-3 hours. Implement T1-09 through T1-13 with separated prompts, one provider, bounded retries, and explicit unavailable outcomes. Verify current provider availability before choosing a model. Run real inference and inspect grounding.
4. Executed notebook, 1-2 hours. Establish a clean Colab run and retain its outputs. Build a simple report/chart if core requirements already pass.

Day 1 gate: Task 1 runs end to end with real data and LLM responses. If it fails, finish it before expanding scope.

## Day 2: evidence, optional scope, delivery

1. Fix remaining Task 1 acceptance failures and complete report, 1-2 hours.
2. Optional Task 3, budget only after Task 1 is complete. Write its spec first. Reuse Task 1 data/sentiment modules; add five tools, autonomous loop, enforced role permissions, typed handoffs, an executed clarification cycle, memory, and JSONL tracing. Allow several hours plus clean notebook execution. Drop optional dashboard work before reducing core verification.
3. Task 2 needs a separate data/training/evaluation budget and a functioning Colab GPU. Attempt it only after a viable full execution path is established; installation, dataset generation, training, merge, evaluation, and upload can consume substantial time. Do not count an unexecuted notebook as completion.
4. Reserve at least 2-3 hours for a clean-environment run, documentation, citations, reflection, public-link checks, personal video recording, and final submission review.

## Work packets for AI assistance

Use this prompt shape for each small implementation step:

> Implement requirements [IDs] from specs/task1-financial.md. First describe the input/output contract and any unclear decision. Implement only this packet using reusable modules. Validate numerical behavior with independently calculated examples where needed. Run the relevant checks and report actual results, failures, and changed files. Update CITATIONS.md with this assistance. Do not invent execution evidence or silently replace live calls with mocks.

Suggested packets:

| Packet | Spec coverage | Completion gate |
|---|---|---|
| 1 | T1-01 to T1-02, T1-08 | Clean market data and truthful summary from live input |
| 2 | T1-03 to T1-06 | Numerical fixtures agree with documented formulas |
| 3 | T1-07 | Ten real unique recent headlines with provenance |
| 4 | T1-09 to T1-13 | Validated live sentiment/signal plus graceful failure demonstrations |
| 5 | T1-14 to T1-15 | Rendered brief and executed clean Colab notebook |
| 6 | SUB-01 to SUB-09 as applicable | Reviewable submission artifacts and access checks |

## Evidence ledger

During implementation, add `requirement ID -> check/result -> artifact path -> run timestamp` entries. Mark each as pending, verified, failed, or not attempted. A software check using mocks and a live demonstration should be recorded separately.

## Personal walkthrough outline: no more than five minutes

- 0:00-0:30: problem, chosen tasks, and input/output.
- 0:30-1:30: run/demo with actual results and notebook outputs.
- 1:30-3:00: architecture, numerical/schema decisions, and key code.
- 3:00-4:15: validation, failures, limitations, and evidence. If multiple tasks, distribute time across their key results.
- 4:15-5:00: what you would improve and where reviewers find artifacts.

The candidate must record and narrate this personally. Practice explaining why Wilder RSI initialization matters, why LLM output needs validation, and what an unavailable-data result means.

## Final gate

Complete the submission checklist in specs/assessment-requirements.md. Use actual outcomes in REFLECTION.md. Inspect Git history and saved outputs for credentials. Verify public links in an incognito/signed-out browser. Do not commit the confidential source assessment PDF. Reply to the assessment email only after reviewing the submission links and personal recording.
