import argparse
import json
import os
import platform
from importlib.metadata import version

from shared.llm import JSONClient, LLMError
from shared.config import load_environment
from task1_financial.schemas import StrictModel


class SmokeResponse(StrictModel):
    status: str


def check(inference=False):
    load_environment()
    report = {"python":platform.python_version(),"groq_key_configured":bool(os.getenv("GROQ_API_KEY")),
              "hf_token_configured":bool(os.getenv("HF_TOKEN")),
              "inference_model":os.getenv("GROQ_MODEL","openai/gpt-oss-20b"),
              "packages":{name:version(name) for name in ["numpy","pandas","pydantic","yfinance","langgraph"]}}
    if inference:
        client = JSONClient()
        try:
            report["inference_smoke"] = client.complete(SmokeResponse,"Return JSON with status='ok'.",{}).model_dump()
        except LLMError as error:
            report["inference_error"] = str(error)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference",action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.inference),indent=2))
