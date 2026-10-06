"""OpenAI-compatible Groq HTTP transport with local schema validation.

JSON-object mode works across a broader set of models than strict schema mode.
Pydantic validation is mandatory before responses reach business logic.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from shared.io import json_safe

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class JSONClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 60, attempts: int = 3, repair_attempts: int = 2,
                 session=None, sleep=time.sleep):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout = timeout
        self.attempts = attempts
        self.repair_attempts = repair_attempts
        self.session = session or requests.Session()
        self.sleep = sleep
        self.calls = 0
        if attempts < 1 or repair_attempts < 1:
            raise ValueError("Retry budgets must be positive")

    def _request(self, messages: list[dict], max_tokens: int) -> str:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not configured; use an environment variable or Colab Secrets")
        for attempt in range(self.attempts):
            try:
                self.calls += 1
                response = self.session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "temperature": 0, "max_completion_tokens": max_tokens,
                          "response_format": {"type": "json_object"}},
                    timeout=self.timeout,
                )
            except requests.RequestException:
                if attempt + 1 == self.attempts:
                    raise LLMError("Inference transport failed after bounded retries") from None
                self.sleep(2 ** attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.attempts:
                    raise LLMError(f"Inference HTTP {response.status_code}; retry budget exhausted")
                # Clamp server advice to keep each retry bounded.
                try:
                    wait = min(30.0, max(0.0, float(response.headers.get("Retry-After", 2 ** attempt))))
                except ValueError:
                    wait = 2 ** attempt
                self.sleep(wait)
                continue
            if response.status_code >= 400:
                # Never include headers, credentials or raw response bodies in logs.
                raise LLMError(f"Inference HTTP {response.status_code}; check model/account configuration")
            try:
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMError("Inference response truncated; increase token budget")
                content = choice["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Missing response content")
                return content
            except (KeyError, IndexError, ValueError, TypeError):
                raise LLMError("Invalid inference response envelope") from None
        raise LLMError("Inference failed")

    def complete(self, schema: type[T], system: str, payload, max_tokens: int = 2048) -> T:
        messages = [
            {"role": "system", "content": system + "\nReturn one JSON object matching this schema: "
             + json.dumps(schema.model_json_schema())},
            {"role": "user", "content": json.dumps(json_safe(payload), allow_nan=False)},
        ]
        for attempt in range(self.repair_attempts):
            content = self._request(messages, max_tokens)
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                # Only safe field/type information is logged, not rejected input.
                details = [{"loc": e["loc"], "type": e["type"]} for e in error.errors()]
                LOGGER.warning("LLM schema validation failed: %s", details)
                if attempt + 1 == self.repair_attempts:
                    raise LLMError("Response schema validation failed after repair budget") from None
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Repair the JSON to satisfy the schema. Errors: "
                                 + json.dumps(details)})
        raise LLMError("Response validation failed")
