---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- peft
- qlora
- synthetic-data
- supply-chain
---

# Supply-chain exception explanation

Draft model card. Fill in actual results before upload.

## Intended use

Assessment demonstration of structured supply-chain exception explanations under explicit input policies. Synthetic examples do not establish performance on operational data. Human review is required before any operational action.

## Training and data

Student: Qwen/Qwen2.5-1.5B-Instruct. Record actual teacher ID, dataset size, industry/topic distribution, 80/10/10 split counts, generation system prompt location, package versions, hardware, and hyperparameters. Note assistant-only loss and group-based splitting.

## Evaluation

Replace with actual base/fine-tuned ROUGE-L, BERTScore, JSON validity, category/severity accuracy, and personal hallucination review. Include sample size and specific failure examples. Do not claim improvement without measured evidence.

## Limitations

Teacher labels and explanations can be wrong despite programmatic checks. Domain correctness is limited to supplied policies. Text-overlap metrics do not establish grounded reasoning. Small held-out samples provide uncertain estimates. Generalization across synthetic industry groups does not prove real-world reliability.
