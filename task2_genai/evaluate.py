from __future__ import annotations

import csv
import gc
import hashlib
import json
from pathlib import Path

from shared.io import utc_now, write_json
from task2_genai.dataset import read_jsonl
from task2_genai.domain import DomainResponse


def infer(model_path, records, max_new_tokens=384):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires the configured Colab GPU")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map={"":0})
    model.eval()
    outputs = []
    try:
        for row in records:
            prompt = tokenizer.apply_chat_template(row["messages"][:-1],tokenize=False,add_generation_prompt=True)
            inputs = tokenizer(prompt,return_tensors="pt",add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs,do_sample=False,max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            outputs.append({"id":row["id"],"text":tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)})
    finally:
        del model
        gc.collect();torch.cuda.empty_cache()
    return outputs


def canonical(text):
    try:
        return json.dumps(json.loads(text),sort_keys=True,ensure_ascii=False)
    except (ValueError,TypeError):
        return text.strip()


def task_accuracy(predictions, records):
    valid = category = severity = 0
    for pred,row in zip(predictions,records,strict=True):
        expected = DomainResponse.model_validate_json(row["messages"][-1]["content"])
        try:
            actual = DomainResponse.model_validate_json(pred["text"])
            valid += 1
            category += actual.exception_type==expected.exception_type
            severity += actual.severity==expected.severity
        except ValueError:
            pass
    return {"json_valid_rate":valid/len(records),"category_accuracy":category/len(records),"severity_accuracy":severity/len(records)}


def evaluate(data_dir="task2_genai/data", merged_dir="task2_genai/models/merged",
             output_dir="task2_genai/outputs", base_model="Qwen/Qwen2.5-1.5B-Instruct"):
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    import numpy as np
    data_path = Path(data_dir)/"test.jsonl"
    records = read_jsonl(data_path)
    if len(records)<10:
        raise ValueError("Need at least ten held-out responses for required review")
    output = Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    write_json(output / "evaluation_protocol.json", {"frozen_at":utc_now(),"test_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "test_count":len(records),"max_new_tokens":384,"do_sample":False,"seed":42,
                "rouge_field":"canonical whole JSON response","bertscore_model":"distilbert-base-uncased",
                "same_prompts_for_both":True,"checkpoint_selection":"validation loss only"})
    references = [canonical(row["messages"][-1]["content"]) for row in records]
    scorer = rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)
    comparison,predictions = {},{}
    for name,path in [("base",base_model),("fine_tuned",merged_dir)]:
        predictions[name] = infer(path,records)
        write_json(output/f"{name}_predictions.json",predictions[name])
        texts = [canonical(row["text"]) for row in predictions[name]]
        rouge = float(np.mean([scorer.score(ref,text)["rougeL"].fmeasure for ref,text in zip(references,texts,strict=True)]))
        _,_,f1 = bert_score(texts,references,lang="en",model_type="distilbert-base-uncased",num_layers=6,
                            device="cpu",batch_size=8,rescale_with_baseline=False)
        comparison[name] = {"rouge_l":rouge,"bertscore_f1":float(f1.mean()),**task_accuracy(predictions[name],records)}
    write_json(output/"comparison.json",comparison)
    lines = ["| Model | ROUGE-L | BERTScore F1 | JSON valid | Category accuracy | Severity accuracy |",
             "|---|---:|---:|---:|---:|---:|"]
    for name,metrics in comparison.items():
        lines.append("| "+name+" | "+" | ".join(f"{metrics[key]:.4f}" for key in
                     ["rouge_l","bertscore_f1","json_valid_rate","category_accuracy","severity_accuracy"])+" |")
    (output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    review = output/"manual_review.csv"
    if not review.exists():
        with review.open("w",newline="",encoding="utf-8") as file:
            writer = csv.DictWriter(file,fieldnames=["id","prompt","reference","prediction","label","reason"])
            writer.writeheader()
            for row,pred in zip(records,predictions["fine_tuned"],strict=True):
                writer.writerow({"id":row["id"],"prompt":row["messages"][1]["content"],
                    "reference":row["messages"][-1]["content"],"prediction":pred["text"],"label":"","reason":""})
    (output/"qualitative_analysis.md").write_text("# Analysis pending personal review\n\nWrite two evidence-based paragraphs: observed improvements with example IDs, then remaining failure modes and next steps. Do not assume improvement before inspecting actual results.\n",encoding="utf-8")
    return comparison


def summarize_review(path="task2_genai/outputs/manual_review.csv"):
    with Path(path).open(encoding="utf-8",newline="") as file:
        rows = list(csv.DictReader(file))
    valid = {"correct","partially_correct","hallucinated"}
    reviewed = [row for row in rows if row["label"] in valid and row["reason"].strip()]
    if len(reviewed)<10:
        raise ValueError("Candidate must personally label and explain at least ten responses")
    if len({row["id"] for row in reviewed})!=len(reviewed):
        raise ValueError("Duplicate review IDs")
    report = {"reviewed":len(reviewed),"hallucinated":sum(row["label"]=="hallucinated" for row in reviewed),
              "hallucination_rate_percent":100*sum(row["label"]=="hallucinated" for row in reviewed)/len(reviewed)}
    write_json(Path(path).parent/"hallucination_review_summary.json",report)
    return report


def publish(model_dir, repo_id):
    import os
    from huggingface_hub import HfApi
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN not configured")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id,repo_type="model",private=False,exist_ok=True)
    api.upload_folder(repo_id=repo_id,folder_path=model_dir,repo_type="model")
    return f"https://huggingface.co/{repo_id}"
