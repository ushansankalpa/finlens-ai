import json

import numpy as np
import pandas as pd
import pytest

from task3_agentic.engine import ResearchSystem
from task3_agentic.schemas import AgentAction, MemoryAnswer
from task3_agentic.tools import annualized_volatility


class FixtureTools:
    def get_price_data(self, ticker, period):
        return {"close":100,"source":"TEST FIXTURE ONLY"}
    def calculate_volatility(self, ticker, window):
        return {"annualized_volatility":.2,"window":window}
    def get_news(self, ticker, n):
        return {"headlines":[{"id":"fixture","title":"Fixture earnings update","source":"TEST FIXTURE",
                    "retrieved_at":"2026-01-01T00:00:00Z","url":None,"published_at":"2026-01-01T00:00:00Z"}]}
    def llm_sentiment(self, headlines):
        return {"score":.2}
    def web_search(self, query):
        raise RuntimeError("Injected search failure")


class ScriptedClient:
    """Only for deterministic control-flow tests; never live assessment evidence."""
    model = "TEST FIXTURE ONLY"
    def complete(self, schema, system, payload, **kwargs):
        observations = payload.get("observations",[])
        successes = [item for item in observations if item["status"]=="ok"]
        ids = [item["id"] for item in successes]
        if schema is MemoryAnswer:
            return schema(answer="Previously retrieved volatility was 0.2.",evidence_ids=ids)
        role = "analyst" if "role analyst" in system else "writer"
        tools = [item["tool"] for item in observations]
        def call(name,args):
            return AgentAction(action="tool",rationale="Fixture test choice",tool_name=name,arguments=args)
        if role=="analyst":
            if payload.get("context",{}).get("headlines") and "llm_sentiment" not in tools:
                return call("llm_sentiment",{"headlines":payload["context"]["headlines"]})
            if "get_price_data" not in tools:
                return call("get_price_data",{"ticker":"AAPL","period":"3y"})
            if "calculate_volatility" not in tools:
                return call("calculate_volatility",{"ticker":"AAPL","window":30})
            return AgentAction(action="finish",rationale="Fixture brief",output={"ticker":"AAPL",
                          "quantitative_summary":"Fixture quantitative summary", "evidence_ids":ids,"limitations":[]})
        if "web_search" not in tools:
            return call("web_search",{"query":"AAPL analyst commentary"})
        if "get_news" not in tools:
            assert observations[-1]["status"]=="error" # Alternative follows an actual failed observation.
            return call("get_news",{"ticker":"AAPL","n":10})
        if not payload["clarification_count"]:
            return AgentAction(action="clarify",rationale="Fixture request",output={
                "question":"Please assess the supplied headline sentiment.","reason":"Clarify direction before reporting."})
        return AgentAction(action="finish",rationale="Fixture complete",output={"ticker":"AAPL",
             "financial_health_summary":"TEST FIXTURE; financial health cannot be established.",
             "financial_health_evidence_ids":ids[:1],"risks":[{"title":f"Fixture risk {i}",
                "explanation":"TEST FIXTURE ONLY","evidence_ids":ids[:1]} for i in range(3)],
             "hedge_strategy_recommendation":{"strategy":"Fixture hedge","rationale":"Fixture volatility",
                "evidence_ids":ids[:1],"limitations":"Not a real research result"},
             "clarification_incorporation":"Incorporated fixture sentiment score.","limitations":["TEST FIXTURE ONLY"]})


def test_multi_agent_fallback_critique_memory_cache_and_log(tmp_path):
    def make():
        return ResearchSystem(client=ScriptedClient(),tools=FixtureTools(),log_dir=tmp_path/"logs",cache_dir=tmp_path/"cache")
    system = make()
    result = system.run("AAPL")
    assert not result["cache_hit"]
    assert system.clarification_count==1
    assert system.tool_calls==5
    assert any(item["kind"]=="clarification_response" for item in system.messages)
    entries = [json.loads(line) for line in (tmp_path/"logs"/"agent_trace.jsonl").read_text().splitlines()]
    assert len(entries)==5
    assert all(len(item["output"])<=200 and item["duration_ms"]>=0 for item in entries)
    assert sum(item["status"]=="error" for item in entries)==1
    answer = system.follow_up("What volatility did you already retrieve?")
    assert answer["additional_tool_calls"]==0
    cached_system = make()
    cached = cached_system.run("AAPL")
    assert cached["cache_hit"] and cached["new_tool_calls"]==0
    assert cached_system.tool_calls==0


def test_role_permissions_and_argument_validation(tmp_path):
    system = ResearchSystem(client=ScriptedClient(),tools=FixtureTools(),log_dir=tmp_path)
    assert system.dispatch("writer","get_price_data",{"ticker":"AAPL","period":"3y"})["status"]=="error"
    assert system.dispatch("analyst","calculate_volatility",{"ticker":"AAPL","window":1})["status"]=="error"
    assert system.tool_calls==2


def test_volatility_independent_calculation():
    returns = np.array([.01,-.02,.03,-.01])
    values = pd.Series(100*np.exp(np.r_[0,np.cumsum(returns)]),index=pd.date_range("2026-01-01",periods=5))
    expected = np.sqrt(sum((returns-returns.mean())**2)/(len(returns)-1))*np.sqrt(252)
    assert annualized_volatility(values,4)["annualized_volatility"]==pytest.approx(expected)
    with pytest.raises(ValueError,match="Need"):
        annualized_volatility(values,5)
