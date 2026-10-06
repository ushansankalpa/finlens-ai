import json

import pytest

from task2_genai.domain import Observation, Policy, Scenario, expected_topic, validate_scenario
from task2_genai.evaluate import summarize_review
from task2_genai.train import tokenize_example


@pytest.mark.parametrize("updates,topic", [({"quality_hold":True},"quality_hold"),
    ({"units_available":0},"stockout"),({"shipment_delay_days":6},"supplier_delay"),
    ({"demand_change_percent":21},"demand_surge"),({"days_of_stock":91},"excess_inventory"),
    ({"quality_hold":None},"insufficient_information")])
def test_domain_policy_oracle(updates,topic):
    record = Observation(**({"units_available":10,"days_of_stock":10,"shipment_delay_days":1,
                            "demand_change_percent":5,"quality_hold":False}|updates))
    policy = Policy(excess_days_threshold=90,delay_days_threshold=5,demand_surge_threshold=20)
    assert expected_topic(record,policy)==topic


def test_policy_priority_and_strict_thresholds():
    record = Observation(units_available=0,days_of_stock=10,shipment_delay_days=99,demand_change_percent=5,quality_hold=True)
    policy = Policy(excess_days_threshold=90,delay_days_threshold=5,demand_surge_threshold=20)
    assert expected_topic(record,policy)=="quality_hold"
    record.units_available=5; record.quality_hold=False; record.shipment_delay_days=5
    with pytest.raises(ValueError,match="No exception"):
        expected_topic(record,policy)


def test_personal_review_cannot_be_empty(tmp_path):
    path = tmp_path/"review.csv"
    path.write_text("id,label,reason\na,,\n",encoding="utf-8")
    with pytest.raises(ValueError,match="personally"):
        summarize_review(path)


class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        return "prompt" if add_generation_prompt else "promptanswer"
    def __call__(self,text,add_special_tokens=False):
        return {"input_ids":[ord(c) for c in text]}


def test_response_masking_and_no_silent_truncation():
    messages = [{"role":"user","content":"x"},{"role":"assistant","content":"y"}]
    example = tokenize_example(FakeTokenizer(),messages,100)
    assert example["labels"][:6]==[-100]*6
    assert example["labels"][6:]==[ord(c) for c in "answer"]
    with pytest.raises(ValueError,match="exceeds"):
        tokenize_example(FakeTokenizer(),messages,5)


def test_teacher_invented_evidence_is_rejected():
    scenario = Scenario(industry="textiles",request="Explain the stockout.",
        record={"units_available":0,"days_of_stock":0,"shipment_delay_days":0,
                "demand_change_percent":0,"quality_hold":False},
        policy={"excess_days_threshold":90,"delay_days_threshold":5,"demand_surge_threshold":20},
        assistant_output={"exception_type":"stockout","severity":"critical","evidence":["units_available=999"],
            "recommended_action":"expedite_replenishment","explanation":"The observation supports a stockout under the explicit policy."})
    with pytest.raises(ValueError,match="evidence"):
        validate_scenario(scenario)
    scenario.assistant_output.evidence=["units_available=0"]
    assert validate_scenario(scenario) is scenario
    scenario.assistant_output.recommended_action="invented_action"
    with pytest.raises(ValueError,match="label/action"):
        validate_scenario(scenario)


def test_group_split_preserves_holdout_and_flags_repeated_requests(tmp_path):
    from task2_genai.dataset import prepare_splits, read_jsonl
    from task2_genai.domain import STUDENT_SYSTEM
    rows=[]
    for i in range(100):
        payload={"industry":f"test-industry-{i//5}","request":"Explain this supply chain exception.",
            "record":{"units_available":i+1,"days_of_stock":100.,"shipment_delay_days":0,
                      "demand_change_percent":0.,"quality_hold":False},
            "policy":{"excess_days_threshold":90.,"delay_days_threshold":5,"demand_surge_threshold":20.}}
        answer={"exception_type":"excess_inventory","severity":"low","evidence":["days_of_stock=100.0"],
                "recommended_action":"pause_reorders","explanation":"Stock coverage exceeds the supplied excess threshold."}
        rows.append({"id":f"fixture-{i}","group_id":f"group-{i//5}","topic":"excess_inventory","fact_hash":str(i),
                     "messages":[{"role":"system","content":STUDENT_SYSTEM},
                                 {"role":"user","content":json.dumps(payload)},
                                 {"role":"assistant","content":json.dumps(answer)}]})
    report=prepare_splits(rows,tmp_path,expected_count=100)
    groups=[]
    for name,expected in [("train",80),("validation",10),("test",10)]:
        split=read_jsonl(tmp_path/f"{name}.jsonl")
        assert len(split)==expected
        groups.append({item["group_id"] for item in split})
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert report["near_duplicate_cross_split_flags"] # Degenerate request wording is detected.
