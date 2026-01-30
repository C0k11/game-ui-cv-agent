import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to import Qwen2VL specific classes if available, otherwise generic
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
except ImportError:
    Qwen2VLForConditionalGeneration = None
    Qwen2VLProcessor = None


class JsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: AutoProcessor, model_name: str, max_length: int = 2048):
        self.data = []
        self.processor = processor
        self.max_length = max_length
        self.model_name = model_name
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.data.append(json.loads(line))
                except Exception:
                    pass
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item.get("image")
        
        # Determine prompt and label
        # This depends on your data format. 
        # Assuming format: {"image": "path", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        # Or simpler: {"image": "path", "prompt": "...", "response": "..."}
        
        prompt_text = item.get("prompt") or "Describe this image."
        response_text = item.get("response") or item.get("label") or ""
        
        if not image_path or not os.path.exists(image_path):
            # Fallback or skip? For simplicity, we might fail or return dummy.
            # Ideally filter these out in __init__.
            # Creating a dummy blank image for robustness if needed, 
            # but better to assume data is clean.
            pass

        # Prepare Qwen2-VL style input
        # Note: Qwen2-VL processor expects a list of messages for chat format usually.
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            }
        ]

        # Use processor to format inputs
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # For training, we need inputs (image + text prompt) and labels (text response).
        # However, standard SFT usually takes the full text and masks the user part.
        # Qwen2VL processor should handle image processing.
        
        # We need to load image
        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            # Create black image
            image = Image.new("RGB", (256, 256), (0, 0, 0))

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Squeeze batch dim
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs["pixel_values"][0] if "pixel_values" in inputs else None
        image_grid_thw = inputs["image_grid_thw"][0] if "image_grid_thw" in inputs else None

        # Create labels: clone input_ids and mask user part?
        # A simple approach for SFT is to train on the whole sequence but ignore padding.
        # Ideally we mask the instruction part.
        # For now, let's just return input_ids as labels (train on prompt too? suboptimal but runs)
        # OR use the data collator to handle masking if we had conversation format parsed.
        labels = input_ids.clone()
        # Mask padding tokens
        labels[attention_mask == 0] = -100 
        
        # TODO: Better masking of user prompt. 
        # For this skeleton, we assume the model trains on the full sequence or we rely on the user to format data correctly.
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            batch["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw
            
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--data_path", required=True, help="Path to jsonl dataset")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=2048)
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # Quantization config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load processor: {e}")
        return

    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    # Target modules usually: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # For Vision models, we might want to target vision encoder too? Usually just LLM part is enough for SFT.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # Qwen2-VL is treated as Causal LM with images
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Dataset
    train_dataset = JsonlDataset(args.data_path, processor, args.model_path, max_length=args.max_len)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True, # Assume Ampere+ GPU (4090)
        optim="paged_adamw_8bit",
        remove_unused_columns=False, # Important for custom VLM inputs
        report_to="tensorboard",
    )
    
    # Data collator
    def collate_fn(examples):
        # We need to stack tensors.
        # input_ids: (batch, seq)
        # attention_mask: (batch, seq)
        # labels: (batch, seq)
        # pixel_values: (batch, c, h, w) or list? Qwen2VL uses flattened pixel_values + image_grid_thw
        
        batch = {}
        for k in examples[0].keys():
            if examples[0][k] is None:
                continue
            if isinstance(examples[0][k], torch.Tensor):
                batch[k] = torch.stack([e[k] for e in examples])
            else:
                batch[k] = [e[k] for e in examples]
        return batch
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
