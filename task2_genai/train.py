"""Explicit QLoRA loop; heavy imports are deferred to the Colab execution path."""
from __future__ import annotations

import gc
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from shared.io import utc_now, write_json
from task2_genai.dataset import read_jsonl


@dataclass
class TrainingConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    rank: int = 16
    alpha: int = 32
    dropout: float = .05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    learning_rate: float = .0002
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation: int = 8
    max_length: int = 1024
    warmup_ratio: float = .1
    weight_decay: float = .01
    max_grad_norm: float = 1
    seed: int = 42


RATIONALE = {
    "model_id": "Small instruct model to fit free GPU memory and preserve a usable prompted baseline",
    "rank": "Moderate adapter capacity for 160 domain-specific training examples",
    "alpha": "Explicit alpha/r scale of two for adapter contribution",
    "dropout": "Regularization for a small synthetic dataset",
    "target_modules": "Attention and MLP adaptation; module existence verified before training",
    "learning_rate": "Starting adapter learning rate; actual held-out loss decides checkpoint selection",
    "epochs": "Three actual loss observations to inspect optimization and generalization",
    "batch_size": "Single-example microbatch limits activation memory",
    "gradient_accumulation": "Effective batch eight on one GPU without eight concurrent sequences",
    "max_length": "Bound memory; reject over-length examples instead of corrupting response JSON",
    "warmup_ratio": "Gradual warmup before cosine decay to reduce early instability",
    "weight_decay": "Explicit regularization of adapter weights",
    "max_grad_norm": "Clip unstable large gradients",
    "seed": "Record reproducible ordering/configuration; hardware kernels may still differ",
    "quantization": "NF4 with double quantization; 4-bit base stays frozen",
    "compute_dtype": "FP16 on T4, BF16 only on a supporting GPU",
    "optimizer": "AdamW, betas=(0.9,0.999), eps=1e-8; explicit standard moment/epsilon choices",
    "scheduler": "Cosine with warmup; smooth decay over the known optimizer-step budget",
    "loss": "Assistant-only causal cross entropy, including assistant EOS; prompts/padding ignored",
    "gradient_checkpointing": "Recompute activations to save memory; disable inference cache during training",
    "packing": "Disabled to keep each policy/example isolated",
    "checkpoint": "Save each epoch and best validation-loss adapter; test set never selects checkpoints",
}


def tokenize_example(tokenizer, messages, max_length):
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not an exact token prefix; verify response masking")
    if len(full_ids) > max_length:
        raise ValueError(f"Example length {len(full_ids)} exceeds max_length {max_length}; increase limit or revise data")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if not any(label != -100 for label in labels):
        raise ValueError("No assistant tokens remain")
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class ResponseCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    def __call__(self, rows):
        import torch
        maximum = max(len(row["input_ids"]) for row in rows)
        result = {key: [] for key in ["input_ids", "attention_mask", "labels"]}
        for row in rows:
            missing = maximum - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.pad_token_id] * missing)
            result["attention_mask"].append(row["attention_mask"] + [0] * missing)
            result["labels"].append(row["labels"] + [-100] * missing)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def cuda_dtype():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Task 2 requires a Colab CUDA GPU; choose a free GPU runtime")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_qlora(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    set_seed(config.seed)
    dtype = cuda_dtype()
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, quantization_config=quantization,
                                                device_map={"": 0}, torch_dtype=dtype)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(config.target_modules).issubset(names):
        raise ValueError("Some configured LoRA target modules do not exist")
    model = get_peft_model(model, LoraConfig(r=config.rank, lora_alpha=config.alpha,
                            lora_dropout=config.dropout, target_modules=config.target_modules,
                            bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, tokenizer, dtype


def smoke_test(config=None):
    import torch
    from task2_genai.domain import STUDENT_SYSTEM
    config = config or TrainingConfig()
    model, tokenizer, dtype = load_qlora(config)
    example = tokenize_example(tokenizer, [{"role":"system", "content":STUDENT_SYSTEM},
            {"role":"user", "content":"A record is missing. Ask for missing data."},
            {"role":"assistant", "content":'{"exception_type":"insufficient_information"}'}], config.max_length)
    batch = {key: value.to("cuda") for key, value in ResponseCollator(tokenizer.pad_token_id)([example]).items()}
    with torch.autocast("cuda", dtype=dtype):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError("NF4 smoke-test loss is nonfinite")
    loss.backward()
    result = {"stage":"smoke_test_only", "loss":float(loss.detach()), "cuda_device":torch.cuda.get_device_name(0),
              "trainable_gradients":sum(p.grad is not None for p in model.parameters() if p.requires_grad)}
    del model, batch, loss
    gc.collect(); torch.cuda.empty_cache()
    return result


def train(data_dir="task2_genai/data", output_dir="task2_genai/checkpoints", config=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    config = config or TrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "training_configuration.json", {"values":asdict(config), "rationale":RATIONALE})
    diversity = json.loads((Path(data_dir) / "diversity_report.json").read_text(encoding="utf-8"))
    if diversity.get("near_duplicate_cross_split_flags"):
        raise ValueError("Resolve cross-split near-duplicate flags before training")
    if config.batch_size < 1 or config.gradient_accumulation < 1 or config.epochs < 1:
        raise ValueError("Batch, accumulation and epoch counts must be positive")
    model, tokenizer, dtype = load_qlora(config)
    rows = {split: read_jsonl(Path(data_dir) / f"{split}.jsonl") for split in ["train", "validation"]}
    if len(rows["train"]) < 80:
        raise ValueError("At least 80 training examples expected")
    tokenized = {split:[tokenize_example(tokenizer,row["messages"],config.max_length) for row in values]
                 for split,values in rows.items()}
    write_json(output / "token_length_report.json", {split:[len(row["input_ids"]) for row in values]
                                                   for split,values in tokenized.items()})
    generator = torch.Generator().manual_seed(config.seed)
    collator = ResponseCollator(tokenizer.pad_token_id)
    loader = DataLoader(tokenized["train"], batch_size=config.batch_size, shuffle=True,
                        generator=generator, collate_fn=collator, num_workers=0)
    validation = DataLoader(tokenized["validation"],batch_size=config.batch_size,collate_fn=collator,num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=config.learning_rate,weight_decay=config.weight_decay,betas=(.9,.999),eps=1e-8)
    total_steps = math.ceil(len(loader)/config.gradient_accumulation)*config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps*config.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype==torch.float16)
    history, best = [], float("inf")
    write_json(output / "hardware.json", {"device":torch.cuda.get_device_name(0),"dtype":str(dtype),
                                           "torch_version":torch.__version__,"started_at":utc_now()})
    try:
        for epoch in range(config.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            train_total = train_tokens = 0
            for step,batch in enumerate(loader):
                batch = {key:value.to("cuda") for key,value in batch.items()}
                with torch.autocast("cuda",dtype=dtype):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss became nonfinite")
                tokens = int((batch["labels"][:,1:]!=-100).sum())
                train_total += float(loss.detach())*tokens; train_tokens += tokens
                group_start = (step//config.gradient_accumulation)*config.gradient_accumulation
                group_size = min(config.gradient_accumulation,len(loader)-group_start)
                scaler.scale(loss/group_size).backward()
                if (step+1)%config.gradient_accumulation==0 or step+1==len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm)
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    if scaler.get_scale() >= previous_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            model.eval(); val_total = val_tokens = 0
            with torch.no_grad():
                for batch in validation:
                    batch = {key:value.to("cuda") for key,value in batch.items()}
                    with torch.autocast("cuda",dtype=dtype):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Validation loss became nonfinite")
                    tokens = int((batch["labels"][:,1:]!=-100).sum())
                    val_total += float(loss)*tokens; val_tokens += tokens
            record = {"epoch":epoch+1,"train_loss":train_total/train_tokens,
                      "validation_loss":val_total/val_tokens,"timestamp":utc_now()}
            history.append(record); print(json.dumps(record))
            write_json(output / "loss_history.json",history)
            model.save_pretrained(output / f"epoch-{epoch+1}")
            if record["validation_loss"] < best:
                best = record["validation_loss"]
                model.save_pretrained(output / "best_adapter"); tokenizer.save_pretrained(output / "best_adapter")
        write_json(output / "loss_assessment.json", {"strictly_decreasing":all(
            history[i]["validation_loss"]<history[i-1]["validation_loss"] for i in range(1,len(history))),
            "first_to_last_improved":history[-1]["validation_loss"]<history[0]["validation_loss"],
            "note":"Actual results; improvement is not guaranteed or fabricated"})
    except torch.cuda.OutOfMemoryError:
        write_json(output / "oom_incident.json", {"timestamp":utc_now(),"configuration":asdict(config),
                    "error":"CUDA out of memory", "suggested_next_attempt":"Reduce max_length only if targets fit; enable/verify checkpointing; restart runtime"})
        raise
    finally:
        del model, optimizer, loader, validation
        gc.collect(); torch.cuda.empty_cache()
    return history


def merge(adapter_dir="task2_genai/checkpoints/best_adapter", output_dir="task2_genai/models/merged", config=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    config = config or TrainingConfig()
    # Reload non-quantized base to avoid exporting quantization-distorted merged weights.
    base = AutoModelForCausalLM.from_pretrained(config.model_id,torch_dtype=torch.float16,device_map="cpu")
    model = PeftModel.from_pretrained(base,adapter_dir)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(output_dir,safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    del merged,model,base
    gc.collect()
    return str(output_dir)
