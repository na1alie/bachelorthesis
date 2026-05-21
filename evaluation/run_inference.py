import argparse
import json
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_FILE = "dataset-instruct-20k/test.jsonl"
N_SAMPLES = 1000
PRED_DIR  = "predictions"

parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True, help="HuggingFace base model ID")
parser.add_argument("--adapter", default=None,  help="Path to LoRA adapter directory; omit for base model")
parser.add_argument("--run",     default="",    help="Optional run name suffix for output/log file names")
args = parser.parse_args()

MODEL_ID     = args.model
MODEL_NAME   = MODEL_ID.split("/")[-1]
RUN_SUFFIX   = f"_{args.run}" if args.run else ""
IS_FINETUNED = args.adapter is not None
MODEL_TAG    = "finetuned" if IS_FINETUNED else "base"

os.makedirs(PRED_DIR, exist_ok=True)

output_path = os.path.join(PRED_DIR, f"{MODEL_NAME}_{MODEL_TAG}{RUN_SUFFIX}.jsonl")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"inference_{MODEL_NAME}_{MODEL_TAG}{RUN_SUFFIX}.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("===== Inference started =====")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Device count: {torch.cuda.device_count()}")
logging.info(f"Model: {MODEL_ID}")
if IS_FINETUNED:
    logging.info(f"Adapter: {args.adapter}")
logging.info(f"Test file: {TEST_FILE}, n_samples: {N_SAMPLES}")
logging.info(f"Output: {output_path}")


#Parsing (from experiment_README)
def parse_output(text):
    triples = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(" | ")
        if len(parts) == 3:
            s, r, o = parts
            triples.append((s, r, o))
    return triples



def load_model():
    logging.info(f"Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    if IS_FINETUNED:
        logging.info(f"Loading adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, samples):
    device = next(model.parameters()).device
    predictions = []

    for sample in samples:
        messages = [m for m in sample["messages"] if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(generated, skip_special_tokens=True)

        gold_triples = []
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                gold_triples = parse_output(msg["content"])
                break

        predictions.append({
            "id":                sample["id"],
            "model_output":        model_output,
            "model_predicted_triples": parse_output(model_output),
            "gold_triples":      gold_triples,
        })

    return predictions


def main():
    logging.info(f"Loading test data ({N_SAMPLES} samples)")
    with open(TEST_FILE) as f:
        samples = [json.loads(l) for l in f][:N_SAMPLES]

    model, tokenizer = load_model()
    logging.info(f"Running inference on {len(samples)} samples...")
    results = run_inference(model, tokenizer, samples)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"Saved {len(results)} predictions → {output_path}")
    logging.info("===== Inference finished =====")


if __name__ == "__main__":
    main()
