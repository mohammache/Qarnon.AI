# MoroccoTone — Fine-tuning Qwen2.5-7B-Instruct on Moroccan Darija
# Run this on Kaggle: Accelerator → GPU T4 x2 (or P100)
# Add your dataset: https://www.kaggle.com/datasets/<you>/moroccoone-darija-instruction-dataset

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# %%capture
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "unsloth", "trl>=0.12.0", "peft", "accelerate",
                "bitsandbytes", "datasets", "--quiet"])

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import os, json, torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ── Cell 3: Config ────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_R         = 16
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.05
BATCH_SIZE     = 2
GRAD_ACCUM     = 4           # effective batch = 2×4 = 8
EPOCHS         = 2
LR             = 2e-4
WARMUP_RATIO   = 0.05
OUTPUT_DIR     = "/kaggle/working/moroccoone-qwen"

# Dataset path (after adding Kaggle dataset to notebook)
DATA_DIR = Path("/kaggle/input/moroccoone-darija-instruction-dataset")
if not DATA_DIR.exists():
    # Fallback: current dir (for local testing)
    DATA_DIR = Path("dataset")

# ── Cell 4: Load model + LoRA ─────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = None,          # auto-detect (bfloat16 on A100, float16 on T4)
    load_in_4bit    = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_R,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM
    random_state        = 42,
)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Cell 5: Load dataset ──────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train_raw = load_jsonl(DATA_DIR / "train.jsonl")
val_raw   = load_jsonl(DATA_DIR / "val.jsonl")

print(f"Train: {len(train_raw):,}  |  Val: {len(val_raw):,}")

# Format messages → single string using Qwen ChatML template
def format_example(example):
    return {"text": tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )}

train_ds = Dataset.from_list(train_raw).map(format_example, remove_columns=["messages"])
val_ds   = Dataset.from_list(val_raw).map(format_example,   remove_columns=["messages"])

# Sanity check
print("\nSample formatted example:")
print(train_ds[0]["text"][:400])

# ── Cell 6: Train ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = train_ds,
    eval_dataset  = val_ds,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    dataset_num_proc   = 2,
    args = TrainingArguments(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate           = LR,
        warmup_steps            = 100,
        lr_scheduler_type       = "cosine",
        fp16                    = not torch.cuda.is_bf16_supported(),
        bf16                    = torch.cuda.is_bf16_supported(),
        logging_steps           = 50,
        eval_strategy           = "steps",
        eval_steps              = 200,
        save_strategy           = "steps",
        save_steps              = 400,
        save_total_limit        = 2,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        optim                   = "adamw_8bit",
        weight_decay            = 0.01,
        seed                    = 42,
        report_to               = "none",
    ),
)

print("\nStarting training...")
trainer_stats = trainer.train()

print(f"\nTraining done.")
print(f"  Runtime : {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"  Loss    : {trainer_stats.metrics['train_loss']:.4f}")

# ── Cell 7: Save LoRA adapter ─────────────────────────────────────────────────
adapter_dir = Path(OUTPUT_DIR) / "lora_adapter"
model.save_pretrained(str(adapter_dir))
tokenizer.save_pretrained(str(adapter_dir))
print(f"\nLoRA adapter saved: {adapter_dir}")
print(f"  Size: {sum(f.stat().st_size for f in adapter_dir.rglob('*') if f.is_file()) / 1e6:.1f} MB")

# ── Cell 8: Quick inference test ──────────────────────────────────────────────
FastLanguageModel.for_inference(model)

test_prompts = [
    "Translate to Darija: I don't have enough time to finish this project.",
    "Write a Moroccan comment reacting to this news: المغرب يتأهل لكأس العالم",
    "What does this Moroccan expression mean? \"راه كيدور عليك\"",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids       = inputs,
            max_new_tokens  = 150,
            temperature     = 0.7,
            do_sample       = True,
            pad_token_id    = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\nPROMPT: {prompt}")
    print(f"OUTPUT: {response}")

# ── Cell 9: Optional — merge + save full model (needs ~15GB disk) ─────────────
# Uncomment if you want a standalone merged model (no adapter dependency):
#
# merged_dir = Path(OUTPUT_DIR) / "merged"
# model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
# print(f"Merged model saved: {merged_dir}")
#
# Or push directly to HuggingFace Hub:
# model.push_to_hub_merged(
#     "your-hf-username/MoroccoTone-Qwen2.5-7B",
#     tokenizer,
#     save_method = "lora",
#     token = "hf_...",
# )
