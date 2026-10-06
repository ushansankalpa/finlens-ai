from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from task1_financial.schemas import StrictModel


class Evidence(StrictModel):
    id: str
    tool: str
    data: dict


class DataBrief(StrictModel):
    ticker: str
    quantitative_summary: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    limitations: list[str]


class AnalystFinish(StrictModel):
    ticker: str
    quantitative_summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    limitations: list[str]


class ClarificationRequest(StrictModel):
    question: str = Field(min_length=10)
    reason: str = Field(min_length=5)


class Risk(StrictModel):
    title: str
    explanation: str
    evidence_ids: list[str] = Field(min_length=1)


class Hedge(StrictModel):
    strategy: str
    rationale: str
    evidence_ids: list[str] = Field(min_length=1)
    limitations: str


class ResearchReport(StrictModel):
    ticker: str
    financial_health_summary: str = Field(min_length=1)
    financial_health_evidence_ids: list[str] = Field(min_length=1)
    risks: list[Risk] = Field(min_length=3, max_length=3)
    hedge_strategy_recommendation: Hedge
    clarification_incorporation: str
    limitations: list[str]


class AgentAction(StrictModel):
    action: Literal["tool", "finish", "clarify"]
    rationale: str = Field(min_length=1)
    tool_name: str | None = None
    arguments: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent(self):
        if self.action == "tool" and not self.tool_name:
            raise ValueError("Tool action needs tool_name")
        if self.action in {"finish", "clarify"} and not self.output:
            raise ValueError("Finish/clarify needs an output object")
        return self


class MemoryAnswer(StrictModel):
    answer: str
    evidence_ids: list[str] = Field(min_length=1)


class AnalystAction(AgentAction):
    output: AnalystFinish | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_output(cls, value):
        if isinstance(value, dict) and value.get("action") == "tool" and value.get("output") == {}:
            value = {**value, "output": None}
        return value


class WriterAction(AgentAction):
    output: ResearchReport | ClarificationRequest | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_tool_output(cls, value):
        if isinstance(value, dict) and value.get("action") == "tool" and value.get("output") == {}:
            value = {**value, "output": None}
        return value


class ToolAction(StrictModel):
    action: Literal["tool"]
    rationale: str
    tool_name: Literal["get_price_data", "get_news", "calculate_volatility", "llm_sentiment", "web_search"]
    arguments: dict
    output: None = None

    @model_validator(mode="before")
    @classmethod
    def empty_output(cls, value):
        if isinstance(value, dict) and value.get("output") == {}:
            return {**value, "output": None}
        return value


class AnalystFinishAction(StrictModel):
    action: Literal["finish"]
    rationale: str
    tool_name: None = None
    arguments: dict = Field(default_factory=dict)
    output: AnalystFinish


class WriterFinishAction(StrictModel):
    action: Literal["finish"]
    rationale: str
    tool_name: None = None
    arguments: dict = Field(default_factory=dict)
    output: ResearchReport


class ClarifyAction(StrictModel):
    action: Literal["clarify"]
    rationale: str
    tool_name: None = None
    arguments: dict = Field(default_factory=dict)
    output: ClarificationRequest


class AnalystDecision(StrictModel):
    decision: ToolAction | AnalystFinishAction


class WriterDecision(StrictModel):
    decision: ToolAction | WriterFinishAction | ClarifyAction
