import json

import pytest
import requests

from shared.llm import JSONClient, LLMError
from task1_financial.schemas import HeadlineSentiment


class Response:
    def __init__(self, status=200, content="{}", headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.content = content
    def json(self):
        return {"choices": [{"finish_reason": "stop", "message": {"content": self.content}}]}


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
    def post(self, *args, **kwargs):
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


VALID = json.dumps({"headline":"Test", "sentiment":"neutral", "confidence":.5, "brief_reason":"No direction"})


def test_validation_repair_and_transient_retry():
    waits = []
    client = JSONClient(api_key="test-only", session=Session([Response(429), Response(content="{bad"),
                         Response(content=VALID)]), sleep=waits.append)
    assert client.complete(HeadlineSentiment, "JSON", {}).sentiment == "neutral"
    assert client.calls == 3
    assert len(waits) == 1


def test_authentication_not_retried_or_secret_logged():
    client = JSONClient(api_key="test-only", session=Session([Response(401)]))
    with pytest.raises(LLMError, match="401") as error:
        client.complete(HeadlineSentiment, "JSON", {})
    assert "test-only" not in str(error.value)
    assert client.calls == 1


def test_malformed_output_exhausts_budget():
    client = JSONClient(api_key="test-only", session=Session([Response(), Response()]))
    with pytest.raises(LLMError, match="schema validation"):
        client.complete(HeadlineSentiment, "JSON", {})


def test_timeout_exhausts_transport_budget():
    client = JSONClient(api_key="test-only", attempts=2,
                       session=Session([requests.Timeout(), requests.Timeout()]), sleep=lambda _: None)
    with pytest.raises(LLMError, match="transport"):
        client.complete(HeadlineSentiment, "JSON", {})
