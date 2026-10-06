from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from shared.io import json_safe, utc_now, write_json
from shared.llm import JSONClient, LLMError
from task1_financial.data import validate_ticker
from task3_agentic.prompts import AGENT_SYSTEM, FOLLOWUP_SYSTEM
from task3_agentic.schemas import AnalystDecision, WriterDecision, AnalystFinish, ClarificationRequest, DataBrief, Evidence, MemoryAnswer, ResearchReport
from task3_agentic.tools import ARG_SCHEMAS, ROLE_TOOLS, FinancialTools


class AgentBudgetError(RuntimeError):
    pass


def compact_for_model(value):
    """Remove bulky duplicate/provenance fields only from the model's request view.

    Full source observations, tool arguments and typed handoffs remain in state/logs.
    Evidence IDs let the report refer back to those authoritative records.
    """
    if isinstance(value, list):
        return [compact_for_model(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"ohlcv", "url", "href", "link"}:
                continue
            if key == "evidence" and isinstance(item, list) and item and isinstance(item[0], dict) and "id" in item[0]:
                result["evidence"] = [{"id": entry["id"], "tool": entry["tool"]} for entry in item]
            else:
                result[key] = compact_for_model(item)
        return result
    return value


class LoopState(TypedDict):
    role: str
    query: str
    context: dict
    steps: int
    action: dict
    result: dict | None


class ResearchSystem:
    def __init__(self, client=None, tools=None, log_dir="task3_agentic/logs",
                 cache_dir="task3_agentic/cache", max_steps=16):
        self.client = client or JSONClient()
        self.tools = tools or FinancialTools(self.client)
        self.log_dir, self.cache_dir = Path(log_dir), Path(cache_dir)
        self.max_steps = max_steps
        self.run_id = uuid.uuid4().hex
        self.observations, self.messages = [], []
        self.clarification_count = 0
        self.tool_calls = 0

    def _trace_message(self, role, kind, payload):
        self.messages.append({"timestamp": utc_now(), "agent": role, "kind": kind,
                              "payload": json_safe(payload)})

    def dispatch(self, role, tool_name, arguments):
        start = time.perf_counter()
        evidence_id = f"{self.run_id[:8]}-T{self.tool_calls + 1:03d}"
        self.tool_calls += 1
        try:
            if tool_name not in ROLE_TOOLS[role]:
                raise PermissionError("Tool not permitted for this role")
            kwargs = ARG_SCHEMAS[tool_name].model_validate(arguments).model_dump()
            if tool_name == "llm_sentiment":
                known = {headline["id"]: headline for observation in self.observations
                         if observation["status"] == "ok" and observation["tool"] == "get_news"
                         for headline in observation["data"].get("headlines", [])}
                requested = kwargs["headlines"]
                if any(item["id"] not in known or item["title"] != known[item["id"]]["title"] for item in requested):
                    raise ValueError("Sentiment needs previously retrieved headline IDs/titles; defer to Writer news handoff")
                kwargs["headlines"] = [known[item["id"]] for item in requested]
            output = json_safe(getattr(self.tools, tool_name)(**kwargs))
            envelope = {"id": evidence_id, "tool": tool_name, "status": "ok", "data": output}
        except Exception as error:
            # Avoid leaking raw provider/authentication errors.
            detail = (str(error) if isinstance(error, (ValidationError, ValueError, PermissionError))
                      else f"{type(error).__name__}: tool unavailable")
            if isinstance(error, ValidationError):
                detail = "Tool argument schema validation failed"
            envelope = {"id": evidence_id, "tool": tool_name, "status": "error", "error": detail}
        duration = (time.perf_counter() - start) * 1000
        entry = {"timestamp": utc_now(), "run_id": self.run_id, "agent": role, "tool_name": tool_name,
                 "inputs": json_safe(arguments), "output": json.dumps(envelope, ensure_ascii=False)[:200],
                 "duration_ms": round(duration, 3), "status": envelope["status"]}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / "agent_trace.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, allow_nan=False) + "\n")
        self.observations.append(envelope)
        self._trace_message(role, "observation", envelope)
        return envelope

    def _allowed_ids(self):
        return {entry["id"] for entry in self.observations if entry["status"] == "ok"}

    def _model_messages(self):
        # Observations are sent once in their own field, never duplicated in history.
        messages = []
        for message in self.messages:
            if message["kind"] == "observation":
                continue
            payload = message["payload"]
            if message["kind"] == "decision":
                payload = {key: payload.get(key) for key in ["action", "tool_name", "rationale"]}
            messages.append({"agent": message["agent"], "kind": message["kind"],
                             "payload": compact_for_model(payload)})
        return messages[-8:]

    def _validate_final(self, role, output):
        if role == "analyst":
            finish = AnalystFinish.model_validate(output)
            if not set(finish.evidence_ids).issubset(self._allowed_ids()):
                raise ValueError("Unknown analyst evidence ID")
            actual = {entry["id"]: entry for entry in self.observations if entry["status"] == "ok"}
            # Copy authoritative observations; never ask the LLM to reproduce numeric dictionaries.
            result = DataBrief(ticker=finish.ticker, quantitative_summary=finish.quantitative_summary,
                evidence=[Evidence(id=id_, tool=actual[id_]["tool"], data=actual[id_]["data"])
                          for id_ in finish.evidence_ids], limitations=finish.limitations)
        else:
            result = ResearchReport.model_validate(output)
            ids = set(result.financial_health_evidence_ids + result.hedge_strategy_recommendation.evidence_ids)
            for risk in result.risks:
                ids.update(risk.evidence_ids)
            if not ids.issubset(self._allowed_ids()):
                raise ValueError("Final report references unobserved evidence")
            if role == "writer" and (not self.clarification_count or not result.clarification_incorporation.strip()):
                raise ValueError("Writer must request and incorporate one clarification before finishing")
        return result.model_dump()

    def _clarify(self, output):
        if self.clarification_count >= 1:
            raise ValueError("Clarification budget already used")
        request = ClarificationRequest.model_validate(output)
        self._trace_message("writer", "clarification_request", request)
        # Typed headline records cross the boundary; Analyst has no news tool.
        headlines = [h for observation in self.observations
                     if observation["status"] == "ok" and observation["tool"] == "get_news"
                     for h in observation["data"].get("headlines", [])]
        response = self._loop("analyst", request.question,
                              {"clarification": request.model_dump(), "headlines": headlines})
        self.clarification_count += 1
        self._trace_message("analyst", "clarification_response", response)
        return {"request": request.model_dump(), "response": response}

    def _loop(self, role, query, context=None):
        action_schema = AnalystDecision if role == "analyst" else WriterDecision
        system = AGENT_SYSTEM.format(role=role, tools=json.dumps({name: ARG_SCHEMAS[name].model_json_schema()
                                        for name in sorted(ROLE_TOOLS[role])}))

        def decide(state):
            if state["steps"] >= self.max_steps:
                raise AgentBudgetError(f"{role} exceeded {self.max_steps} decisions; see saved trace")
            response = self.client.complete(action_schema, system, {"query": query, "context": compact_for_model(state["context"]),
                         "observations": compact_for_model(self.observations), "recent_messages": self._model_messages(),
                         "clarification_count": self.clarification_count}, max_tokens=2048)
            action = getattr(response, "decision", response)
            self._trace_message(role, "decision", action)
            return {"action": action.model_dump(), "steps": state["steps"] + 1}

        def execute(state):
            action = action_schema.model_validate({"decision": state["action"]}).decision
            context = dict(state["context"])
            try:
                if action.action == "tool":
                    self.dispatch(role, action.tool_name, action.arguments)
                elif action.action == "clarify":
                    if role != "writer":
                        raise ValueError("Only Writer can request clarification")
                    context["clarification_handoff"] = self._clarify(action.output)
                else:
                    result = self._validate_final(role, action.output)
                    self._trace_message(role, "structured_output", result)
                    return {"result": result}
            except (ValidationError, ValueError) as error:
                context["last_validation_error"] = ("Output schema validation failed" if isinstance(error, ValidationError)
                                                    else str(error))
                self._trace_message(role, "validation_error", context["last_validation_error"])
            return {"context": context, "result": None}

        graph = StateGraph(LoopState)
        graph.add_node("decide", decide)
        graph.add_node("execute", execute)
        graph.add_edge(START, "decide")
        graph.add_edge("decide", "execute")
        graph.add_conditional_edges("execute", lambda state: "done" if state.get("result") else "continue",
                                    {"done": END, "continue": "decide"})
        result = graph.compile().invoke({"role": role, "query": query, "context": context or {},
                         "steps": 0, "action": {}, "result": None},
                         {"recursion_limit": self.max_steps * 3 + 5})
        return result["result"]

    def run(self, ticker="AAPL", mode="multi", refresh=False):
        ticker = validate_ticker(ticker)
        if mode not in {"single", "multi"}:
            raise ValueError("Mode must be single or multi")
        config = hashlib.sha256(json.dumps({"schema": 4, "model": self.client.model,
                                            "mode": mode, "max_steps": self.max_steps}, sort_keys=True).encode()).hexdigest()[:12]
        today = datetime.now(timezone.utc).date().isoformat()
        cache = self.cache_dir / ticker / f"{today}-{config}.json"
        if cache.exists() and not refresh:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            ResearchReport.model_validate(payload["report"])
            self.observations = payload["observations"]
            self.messages = payload["messages"]
            return {**payload, "cache_hit": True, "new_tool_calls": 0}
        # A fresh query must not inherit an earlier ticker's observations or critique count.
        self.run_id = uuid.uuid4().hex
        self.observations, self.messages = [], []
        self.clarification_count = self.tool_calls = 0
        if hasattr(self.tools, "market_cache"):
            self.tools.market_cache.clear()
        query = (f"Analyse the current financial health and market sentiment of {ticker}. Identify the top three "
                 "risks to its share price over the next 90 days and suggest one data-driven hedge strategy.")
        try:
            if mode == "single":
                report = self._loop("researcher", query)
            else:
                brief = DataBrief.model_validate(self._loop("analyst", query))
                self._trace_message("analyst", "handoff_to_writer", brief)
                report = self._loop("writer", query, {"data_brief": brief.model_dump()})
            if report["ticker"].upper() != ticker:
                raise ValueError("Report ticker does not match query")
            payload = {"ticker": ticker, "date": today, "config": config, "report": report,
                       "observations": self.observations, "messages": self.messages,
                       "cache_hit": False, "new_tool_calls": self.tool_calls}
            write_json(cache, payload)
            write_json(self.log_dir / f"{mode}_research.json", payload)
            return payload
        finally:
            write_json(self.log_dir / f"{self.run_id}_messages.json", self.messages)

    def follow_up(self, question):
        before = self.tool_calls
        answer = self.client.complete(MemoryAnswer, FOLLOWUP_SYSTEM,
                     {"question": question, "observations": compact_for_model(self.observations), "messages": self._model_messages()})
        if not set(answer.evidence_ids).issubset(self._allowed_ids()):
            raise ValueError("Memory response cites unobserved evidence")
        return {"answer": answer.model_dump(), "additional_tool_calls": self.tool_calls - before}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--model", help="Optional Groq model override for agent decisions")
    args = parser.parse_args()
    system = ResearchSystem(client=JSONClient(model=args.model))
    try:
        result = system.run(args.ticker, args.mode, args.refresh)
    except (LLMError, AgentBudgetError, ValueError) as error:
        parser.exit(1, f"Research run incomplete: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
