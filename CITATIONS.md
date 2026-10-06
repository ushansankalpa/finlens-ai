# Citations and AI assistance

Record each distinct assistance instance and its affected files. Use the assessment's required format, including exact model identity when known, prompt, and date. Preserve full teacher-generation system prompts if Task 2 is attempted. Cite adapted external code with URL, file, and original line range. Do not claim a source was used unless it was actually used.

## Planning assistance

AI-ASSISTED: OpenAI Codex (GPT-6; exact serving model identifier not exposed in this session), Prompt: 'Read the assessment and guide me to complete the assignment in a spec-driven way', Date: 2026-10-06

Scope: assessment interpretation and drafts of README.md, PLAN.md, specs/assessment-requirements.md, specs/task1-financial.md, specs/task2-genai.md, specs/task3-agentic.md, CITATIONS.md, and REFLECTION.md. Prompt text above summarizes the user's request; the original request also contained the assessment email. Follow-up scope instruction: 'All three tasks', with time response '24 h'. No implementation, numerical results, training metrics, or executed notebook outputs were generated.

## Implementation assistance

AI-ASSISTED: OpenAI Codex (GPT-6; exact serving model identifier not exposed in this session), Prompt: 'start to implementation', Date: 2026-10-06

Scope: initial Python implementation in shared/, task1_financial/, task2_genai/, task3_agentic/, tests/, and scripts/; Colab notebook authoring; dependency configuration; setup, task README and evidence documentation. User clarified that Groq/Colab setup help is needed and requested project title 'CDAZZDEV — ML Technical'. Tests use clearly labeled fixtures; saved AAPL data came from live yfinance/RSS retrieval. No live LLM responses, synthetic teacher dataset, training metrics, or manual response-review labels were fabricated. Candidate review and understanding remain necessary.

## Documentation consulted during implementation

These primary sources were consulted for API behavior and design; no repository code snippets were copied:

- https://console.groq.com/docs/structured-outputs - JSON-object versus schema modes and local validation needs.
- https://console.groq.com/docs/rate-limits - bounded retry and account-specific availability.
- https://docs.langchain.com/oss/python/langgraph/graph-api - StateGraph nodes, routing, and state.
- https://huggingface.co/docs/transformers/quantization/bitsandbytes - NF4/double quantization settings.
- https://huggingface.co/docs/peft/developer_guides/quantization - preparation of a quantized base for adapter training.
- https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md - adapter merge API.
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct - student model and tokenizer.
- https://research.google.com/colaboratory/faq.html - free GPU availability limitations.

## Adapted open-source code

None at this planning stage. Ordinary dependencies should be listed in project documentation; adapted snippets additionally need the required source citation.
