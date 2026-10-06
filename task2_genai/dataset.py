from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from shared.io import utc_now, write_json
from shared.llm import JSONClient, LLMError
from task2_genai.domain import (STUDENT_SYSTEM, TEACHER_SYSTEM, TOPICS, Scenario,
                                user_payload, validate_scenario)

INDUSTRIES = ["semiconductor", "textiles", "automotive", "agriculture", "consumer electronics",
              "pharmaceutical packaging", "construction materials", "food distribution", "furniture",
              "solar equipment", "marine parts", "paper production", "cosmetics", "sports equipment",
              "industrial machinery", "pet supplies", "bicycle manufacturing", "glass production",
              "water treatment", "aviation components"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records), encoding="utf-8")


def generate(output_dir="task2_genai/data", client=None, count=200, seed=42):
    if count < 100 or count % 20:
        raise ValueError("Count must be at least 100 and divisible by 20 for grouped 80/10/10 splitting")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    client = client or JSONClient(model=os.getenv("TEACHER_MODEL", "openai/gpt-oss-120b"))
    if client.model == "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("Teacher and student must differ")
    prompt_file = directory / "teacher_system_prompt.txt"
    if prompt_file.exists() and prompt_file.read_text(encoding="utf-8") != TEACHER_SYSTEM:
        raise ValueError("Existing generation used another prompt; use a new data directory")
    prompt_file.write_text(TEACHER_SYSTEM, encoding="utf-8")
    path = directory / "generated.jsonl"
    records = read_jsonl(path) if path.exists() else []
    seen = {item["fingerprint"] for item in records}
    existing_ids = {item["id"] for item in records}
    per_group = count // 20
    for index in range(count):
        id_ = f"scenario-{index:04d}"
        if id_ in existing_ids:
            continue
        group = index // per_group
        topic = TOPICS[(index % per_group + group) % len(TOPICS)]
        for attempt in range(4):
            try:
                scenario = client.complete(Scenario, TEACHER_SYSTEM,
                    {"id": id_, "topic": topic, "industry": INDUSTRIES[group],
                     "variation_seed": seed + index * 11 + attempt,
                     "length_style": ["concise", "moderate", "detailed"][index % 3],
                     "boundary_case": index % 4 == 0,
                     "avoid_request_wordings": [json.loads(row["messages"][1]["content"])["request"]
                                                 for row in records[-10:]]}, max_tokens=2400)
                validate_scenario(scenario)
                if scenario.assistant_output.exception_type != topic or scenario.industry.casefold() != INDUSTRIES[group].casefold():
                    raise ValueError("Teacher did not follow requested topic/industry")
                payload = user_payload(scenario)
                user = json.dumps(payload, sort_keys=True)
                fingerprint = hashlib.sha256(user.encode()).hexdigest()
                # Also deduplicate the structured facts independently of request wording.
                fact_hash = hashlib.sha256(json.dumps({"record": payload["record"], "policy": payload["policy"]}, sort_keys=True).encode()).hexdigest()
                if fingerprint in seen or any(item["fact_hash"] == fact_hash for item in records):
                    raise ValueError("Duplicate teacher scenario")
                item = {"id": id_, "group_id": f"industry-{group:02d}", "topic": topic,
                        "fingerprint": fingerprint, "fact_hash": fact_hash,
                        "teacher_model": client.model, "generated_at": utc_now(),
                        "messages": [{"role": "system", "content": STUDENT_SYSTEM},
                                     {"role": "user", "content": user},
                                     {"role": "assistant", "content": scenario.assistant_output.model_dump_json()}]}
                with path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(item, ensure_ascii=False) + "\n")
                records.append(item)
                seen.add(fingerprint)
                print(f"Generated {len(records)}/{count}: {topic}")
                break
            except (ValueError, LLMError) as error:
                if attempt == 3:
                    raise RuntimeError(f"Could not generate valid {id_}; existing progress saved: {error}") from None
    return prepare_splits(records, directory, seed, expected_count=count)


def prepare_splits(records, directory, seed=42, expected_count=None):
    if expected_count is not None and len(records) != expected_count:
        raise ValueError("Unexpected dataset size; use a fresh directory when changing count")
    if len(records) < 100:
        raise ValueError("At least 100 validated examples required")
    if len({item["id"] for item in records}) != len(records):
        raise ValueError("Duplicate scenario IDs")
    for item in records:
        messages = item["messages"]
        if [message["role"] for message in messages] != ["system", "user", "assistant"]:
            raise ValueError("Every example needs system/user/assistant turns")
        if messages[0]["content"] != STUDENT_SYSTEM:
            raise ValueError("Unexpected student system prompt")
        payload = json.loads(messages[1]["content"])
        scenario = Scenario(**payload, assistant_output=json.loads(messages[2]["content"]))
        validate_scenario(scenario)
        if scenario.assistant_output.exception_type != item["topic"]:
            raise ValueError("Topic metadata does not match example label")
    directory = Path(directory)
    groups = sorted({record["group_id"] for record in records})
    if len(groups) != 20:
        raise ValueError("Expected twenty industry groups")
    sizes = Counter(item["group_id"] for item in records)
    if len(set(sizes.values())) != 1:
        raise ValueError("Groups must have equal size to preserve exact split ratios")
    hashes = [item["fact_hash"] for item in records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Duplicate structured facts found")
    random.Random(seed).shuffle(groups)
    assignments = {"train": set(groups[:16]), "validation": set(groups[16:18]), "test": set(groups[18:])}
    report = {"seed": seed, "generated_count": len(records), "splits": {},
              "keyword_counts": {}, "near_duplicate_cross_split_flags": [],
              "note": "Industry-group split reduces leakage; lexical overlap flags still require inspection."}
    split_records = {}
    for split, group_ids in assignments.items():
        rows = sorted([item for item in records if item["group_id"] in group_ids], key=lambda item: item["id"])
        write_jsonl(directory / f"{split}.jsonl", rows)
        split_records[split] = rows
        lengths = [len(item["messages"][1]["content"]) for item in rows]
        report["splits"][split] = {"count": len(rows), "groups": sorted(group_ids),
            "topic_counts": dict(Counter(item["topic"] for item in rows)),
            "prompt_char_lengths": lengths, "length_quantiles": dict(zip(
                ["min", "p25", "median", "p75", "max"], np.quantile(lengths, [0,.25,.5,.75,1]).tolist()))}
    full_text = " ".join(item["messages"][1]["content"].lower() for item in records)
    report["keyword_counts"] = {word: full_text.count(word) for word in ["stock", "delay", "quality", "demand", "null"]}
    # Facts must never duplicate across splits. Similar requests need manual investigation.
    import re
    def request_tokens(row):
        payload = json.loads(row["messages"][1]["content"])
        return set(re.findall(r"[a-z]+", payload["request"].lower()))
    for left in ["validation", "test"]:
        for item in split_records[left]:
            tokens = request_tokens(item)
            for other in split_records["train"]:
                comparison = request_tokens(other)
                overlap = len(tokens & comparison) / max(1, len(tokens | comparison))
                if overlap > .85:
                    report["near_duplicate_cross_split_flags"].append({"left": item["id"], "right": other["id"], "jaccard": overlap})
    report["split_manifest_hash"] = hashlib.sha256(json.dumps(assignments, default=sorted, sort_keys=True).encode()).hexdigest()
    write_json(directory / "diversity_report.json", report)
    if report["near_duplicate_cross_split_flags"]:
        print("WARNING: cross-split near-duplicate requests flagged; inspect/regenerate before training")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="task2_genai/data")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, count=args.count), indent=2))
