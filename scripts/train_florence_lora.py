import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:
    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        setattr(PreTrainedModel, "_supports_sdpa", False)
except Exception:
    pass

try:
    from transformers.generation.utils import GenerationMixin

    if not hasattr(GenerationMixin, "_supports_sdpa"):
        setattr(GenerationMixin, "_supports_sdpa", False)
except Exception:
    pass


class FlorenceJsonlDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class FlorenceCollator:
    def __init__(self, processor, max_target_length: int):
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [Image.open(item["image"]).convert("RGB") for item in batch]
        prompts = [str(item["prompt"]) for item in batch]
        targets = [str(item["target_text"]) for item in batch]
        model_inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        labels = self.processor.tokenizer(
            text=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def _load_records(path: Path, max_samples: int, seed: int, val_ratio: float):
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("image") and obj.get("prompt") and obj.get("target_text"):
            rows.append(obj)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        return [], []
    val_count = int(round(len(rows) * max(0.0, min(val_ratio, 0.5))))
    if val_count <= 0:
        return rows, []
    return rows[val_count:], rows[:val_count]


def _pick_dtype(device: str):
    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _find_target_modules(model) -> List[str]:
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "fc1", "fc2"}
    found = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in wanted and isinstance(module, torch.nn.Linear):
            found.add(leaf)
    if found:
        return sorted(found)
    fallback = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if isinstance(module, torch.nn.Linear) and any(tok in name.lower() for tok in ("attn", "attention", "encoder", "decoder")):
            fallback.add(leaf)
    return sorted(fallback)[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=r"D:\Project\ai game secretary\data\florence_ui_dataset\florence_lora.jsonl")
    ap.add_argument("--output-dir", default=r"D:\Project\ml_cache\models\florence_ui_lora")
    ap.add_argument("--model-id", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--cache-dir", default=r"D:\Project\ml_cache\models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--train-batch-size", type=int, default=1)
    ap.add_argument("--eval-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-target-length", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, val_rows = _load_records(dataset_path, args.max_samples, args.seed, args.val_ratio)
    if not train_rows:
        raise RuntimeError(f"No training rows found in {dataset_path}")

    dtype = _pick_dtype(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, cache_dir=args.cache_dir)

    load_attempts = [
        {
            "trust_remote_code": True,
            "dtype": dtype,
            "attn_implementation": "eager",
            "cache_dir": args.cache_dir,
        },
        {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "attn_implementation": "eager",
            "cache_dir": args.cache_dir,
        },
        {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "cache_dir": args.cache_dir,
        },
    ]
    model = None
    last_err: Optional[Exception] = None
    for kwargs in load_attempts:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
            last_err = None
            break
        except TypeError as e:
            last_err = e
    if model is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError("Failed to load Florence model")

    if str(args.device).lower().startswith("cuda") and torch.cuda.is_available():
        model = model.to(args.device)
    target_modules = _find_target_modules(model)
    if not target_modules:
        raise RuntimeError("Could not find LoRA target modules for Florence model")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    try:
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False
    except Exception:
        pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    train_ds = FlorenceJsonlDataset(train_rows)
    eval_ds = FlorenceJsonlDataset(val_rows) if val_rows else None
    collator = FlorenceCollator(processor, args.max_target_length)

    train_args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "remove_unused_columns": False,
        "fp16": bool(str(args.device).lower().startswith("cuda") and torch.cuda.is_available()),
        "report_to": [],
        "dataloader_num_workers": 0,
        "seed": args.seed,
    }
    ta_sig = inspect.signature(TrainingArguments.__init__)
    eval_value = "epoch" if eval_ds is not None else "no"
    if "evaluation_strategy" in ta_sig.parameters:
        train_args_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in ta_sig.parameters:
        train_args_kwargs["eval_strategy"] = eval_value
    train_args = TrainingArguments(**train_args_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path.resolve()),
                "train_rows": len(train_rows),
                "eval_rows": len(val_rows),
                "target_modules": target_modules,
                "model_id": args.model_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
