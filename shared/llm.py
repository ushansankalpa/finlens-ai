"""OpenAI-compatible Groq HTTP transport with local schema validation.

Known GPT-OSS models use best-effort JSON schema; other models use JSON-object mode.
Pydantic validation is mandatory before responses reach business logic.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from shared.io import json_safe
from shared.config import load_environment

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class RejectedOutput(LLMError):
    def __init__(self, code, content):
        super().__init__(f"Provider rejected generated JSON ({code})")
        self.code = code
        self.content = content


class JSONClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 60, attempts: int = 3, repair_attempts: int = 2,
                 session=None, sleep=time.sleep):
        load_environment()
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

    def _request(self, messages: list[dict], max_tokens: int, schema=None) -> str:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not configured; use an environment variable or Colab Secrets")
        for attempt in range(self.attempts):
            response_format = {"type": "json_object"}
            if schema is not None and self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
                response_format = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "strict": False, "schema": schema.model_json_schema()}}
            body = {"model": self.model, "messages": messages, "temperature": 0,
                    "max_completion_tokens": max_tokens, "response_format": response_format}
            if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
                body["reasoning_effort"] = "low"
            try:
                self.calls += 1
                response = self.session.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
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
                # Surface only the provider's error code/message, never raw bodies or headers.
                try:
                    error = response.json().get("error", {})
                    detail = str(error.get("message", "")) if isinstance(error, dict) else ""
                    code = str(error.get("code", "")) if isinstance(error, dict) else ""
                    failed = error.get("failed_generation", "") if isinstance(error, dict) else ""
                except (ValueError, TypeError, AttributeError):
                    detail, code = "", ""
                detail = detail.replace(self.api_key, "[REDACTED]")
                detail = re.sub(r"(?:gsk_|hf_)[A-Za-z0-9]+", "[REDACTED]", detail)[:500]
                if response.status_code == 400 and code in {"json_validate_failed", "output_parse_failed"}:
                    raise RejectedOutput(code, failed if isinstance(failed, str) else json.dumps(failed))
                raise LLMError(f"Inference HTTP {response.status_code} ({code or 'request_error'}): "
                               f"{detail or 'check model/account configuration'}")
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
            try:
                content = self._request(messages, max_tokens, schema)
            except RejectedOutput as error:
                LOGGER.warning("Provider JSON generation rejected: %s", error.code)
                if attempt + 1 == self.repair_attempts:
                    raise LLMError("Provider JSON generation failed after repair budget") from None
                messages.append({"role": "assistant", "content": error.content or "{}"})
                messages.append({"role": "user", "content": "The provider rejected your generated JSON. Return a complete object conforming exactly to the supplied schema, including required nested fields; no commentary or native tool-call syntax."})
                continue
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                # Only safe field/type information is logged, not rejected input.
                details = [{"loc": e["loc"], "type": e["type"], "message": e["msg"]} for e in error.errors()]
                LOGGER.warning("LLM schema validation failed: %s", details)
                if attempt + 1 == self.repair_attempts:
                    raise LLMError("Response schema validation failed after repair budget") from None
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Repair the JSON to satisfy the schema. Errors: "
                                 + json.dumps(details)})
        raise LLMError("Response validation failed")
