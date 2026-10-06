"""Verify response masking with the real base-model tokenizer; no GPU needed."""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HOME"] = str(ROOT/".hf-cache")


def main():
    from transformers import AutoTokenizer
    from task2_genai.domain import STUDENT_SYSTEM
    from task2_genai.train import tokenize_example
    from shared.io import write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline",action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",local_files_only=args.offline)
    messages = [{"role":"system","content":STUDENT_SYSTEM},
        {"role":"user","content":"Record is missing essential facts. Ask for the missing record."},
        {"role":"assistant","content":'{"exception_type":"insufficient_information","severity":"unknown","evidence":["units_available=None"],"recommended_action":"request_missing_data","explanation":"Essential facts are missing, so no supported classification is possible."}'}]
    example = tokenize_example(tokenizer,messages,1024)
    response_ids = [id_ for id_ in example["labels"] if id_!=-100]
    decoded = tokenizer.decode(response_ids,skip_special_tokens=False)
    if not decoded.startswith(messages[-1]["content"]):
        raise RuntimeError("Response masking does not recover the complete assistant JSON")
    result = {"execution_stage":"real_tokenizer_only_no_training", "model":"Qwen/Qwen2.5-1.5B-Instruct",
              "total_tokens":len(example["input_ids"]),"response_tokens":len(response_ids),
              "complete_response_preserved":True,"decoded_response":decoded}
    write_json(ROOT/"task2_genai"/"outputs"/"tokenizer_check.json",result)
    print(result)


if __name__=="__main__":
    main()
