import argparse
import torch
import os
import json
import logging
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# =====================================================
# 0. Arguments & Paths
# =====================================================
DATASET_DIRS = {
    "rebel":      "data_rebel/dataset-instruct-20k",
    "rebel-full": "data_rebel/dataset-instruct-full-20k",
    "lagrange": "data_lagrange/dataset-instruct-20k",
    "tekgen":   "data_tekgen/dataset-instruct-20k",
    "tum_full":    "data_tum/dataset-instruct-20k-full",
    "tum_p99":     "data_tum/dataset-instruct-20k-p99",
    "tum_p99_new": "data_tum/dataset-instruct-large-p99_new",
    "tum_e3":      "data_tum/dataset-instruct-20k-e3",
    "wikinre":     "data_wikinre/dataset-instruct-20k",
}

parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True, help="HuggingFace model ID")
parser.add_argument("--dataset", choices=["rebel", "rebel-full", "lagrange", "tekgen", "tum_full", "tum_p99", "tum_p99_new", "tum_e3", "wikinre"], default="rebel")
parser.add_argument("--run",     default="", help="Optional run name suffix for log file")
parser.add_argument("--lr",      type=float, default=2e-4, help="Learning rate")
parser.add_argument("--lora_r",  type=int,   default=16,   help="LoRA rank")
args = parser.parse_args()

MODEL_ID    = args.model
MODEL_NAME  = MODEL_ID.split("/")[-1]
RUN_SUFFIX  = f"_{args.run}" if args.run else ""
OUTPUT_DIR  = f"./{MODEL_NAME}-{args.dataset}-qlora{RUN_SUFFIX}"
DATASET_DIR = DATASET_DIRS[args.dataset]
MAX_SEQ_LENGTH = 1024


# =====================================================
# 1. Logging Setup (for Slurm)
# =====================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"train_{MODEL_NAME}_{args.dataset}{RUN_SUFFIX}.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

run = wandb.init(
    entity="ge94say-technical-university-of-munich",
    project="bachelorthesis-re",
    name=f"{MODEL_NAME}_{args.dataset}{RUN_SUFFIX}",
    config={
        "model":        MODEL_ID,
        "dataset":      args.dataset,
        "run":          args.run,
        "epochs":       8,
        "learning_rate": args.lr,
        "batch_size":   8,
        "lora_r":       args.lora_r,
        "lora_alpha":   args.lora_r * 2,
    },
)

logging.info("===== Training started =====")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Device count: {torch.cuda.device_count()}")

logging.info(f"Loading model: {MODEL_ID}")


# =====================================================
# 2. LoRA Configuration
# =====================================================
peft_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_r * 2,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)
logging.info("LoRA configuration initialized.")


# =====================================================
# 3. Load tokenizer and model
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

logging.info("Tokenizer loaded.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False  

logging.info("Model loaded and gradient checkpointing enabled.")


# =====================================================
# 4. Load dataset
# =====================================================
logging.info(f"Loading dataset from {DATASET_DIR}")

dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATASET_DIR, "train.jsonl"),
        "valid": os.path.join(DATASET_DIR, "valid.jsonl"),
    }
)

logging.info(f"Train samples: {len(dataset['train'])}")
logging.info(f"Valid samples: {len(dataset['valid'])}")


# =====================================================
# 5. Formatting function (convert messages → text)
# =====================================================
def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

logging.info("Applying formatting function...")
dataset = dataset.map(formatting_func)
logging.info("Formatting complete.")


# =====================================================
# 6. TrainingArguments
# =====================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_torch_fused",
    learning_rate=args.lr,
    bf16=True,
    logging_steps=50,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=8,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="wandb",
    dataset_text_field="text",
    max_length=MAX_SEQ_LENGTH,
)

logging.info("SFTConfig initialized.")


# =====================================================
# 7. Trainer
# =====================================================
logging.info("Initializing SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    peft_config=peft_config,
    processing_class=tokenizer,
)

logging.info("Trainer created, starting training...")


# =====================================================
# 8. Training
# =====================================================
trainer.train()

logging.info("Training completed.")


# =====================================================
# 9. Save outputs
# =====================================================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "training_args.json"), "w") as f:
    json.dump(training_args.to_dict(), f, indent=2)

logging.info("Model and tokenizer saved.")

with open(os.path.join(OUTPUT_DIR, "loss_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

logging.info("Loss history saved.")

run.finish()
logging.info("===== Training finished successfully =====")
