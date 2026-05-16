import argparse
import json
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="HuggingFace model ID")
args = parser.parse_args()

MODEL_ID   = args.model
MODEL_NAME = MODEL_ID.split("/")[-1]
TEST_FILE  = "dataset-instruct-20k/test.jsonl"
N_SAMPLES  = 1000

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"evaluate_base_{MODEL_NAME}.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("===== Evaluation started =====")
logging.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logging.info(f"Current GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Device count: {torch.cuda.device_count()}")
logging.info(f"Model: {MODEL_ID}")
logging.info(f"Test file: {TEST_FILE}, n_samples: {N_SAMPLES}")



#Parsing & Normalization (from experiment_README)

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


def normalize(text):
    return text.strip().lower()


def normalize_triple(s, r, o):
    return (normalize(s), normalize(r), normalize(o))

###############################
#Evaluation of three evaluation types (from experiment_README)
#comparison predictions and gold -> calculation recall, precision, f1 

#Strict Match: A predicted triple `(s, r, o)` is correct only if all three elements exactly match the gold triple after normalization.
def evaluate_strict(predictions, gold_list):
    tp, fp, fn = 0, 0, 0
    for pred_triples, gold_triples in zip(predictions, gold_list):
        pred_set = set(normalize_triple(*t) for t in pred_triples)
        gold_set = set(normalize_triple(*t) for t in gold_triples)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


#Boundaries Match: Entity boundaries (subject and object spans) must be correct. Relation type is not required to match.
def evaluate_boundaries(predictions, gold_list):
    tp, fp, fn = 0, 0, 0
    for pred_triples, gold_triples in zip(predictions, gold_list):
        pred_set = set((normalize(s), normalize(o)) for s,_,o in pred_triples)
        gold_set = set((normalize(s), normalize(o)) for s,_,o in gold_triples)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


#Partial Match:Subject and object only need to partially match the gold spans (substring match).
def evaluate_partial(predictions, gold_list):
    tp, fp, fn = 0, 0, 0
    for pred_triples, gold_triples in zip(predictions, gold_list):
        matched_gold = set()
        for ps, pr, po in pred_triples:
            matched = False
            for gi, (gs, gr, go) in enumerate(gold_triples):
                if gi not in matched_gold and normalize(pr) == normalize(gr):
                    if (normalize(ps) in normalize(gs) or normalize(gs) in normalize(ps)) and (normalize(po) in normalize(go) or normalize(go) in normalize(po)):
                        tp += 1
                        matched_gold.add(gi)
                        matched = True
                        break
            if not matched:
                fp += 1
        fn += len(gold_triples) - len(matched_gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


############################

#get predictions with model

def load_model():
    logging.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, samples, max_new_tokens=256):
    predictions = []
    device = next(model.parameters()).device

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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(parse_output(text))

    return predictions


###################################

def main():
    logging.info(f"Loading test data ({N_SAMPLES} samples)")
    with open(TEST_FILE) as f:
        samples = [json.loads(l) for l in f][:N_SAMPLES]

    gold_list = []
    for sample in samples:
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                gold_list.append(parse_output(msg["content"]))
                break

    model, tokenizer = load_model()

    logging.info(f"Running inference on {len(samples)} samples...")
    predictions = run_inference(model, tokenizer, samples)

    #calcutate precision, recall and f1 with all three evaluation techniques
    p, r, f1 = evaluate_strict(predictions, gold_list)
    logging.info("Strict Match:")
    logging.info(f"Precision: {p:.4f}")
    logging.info(f"Recall:    {r:.4f}")
    logging.info(f"F1:        {f1:.4f}")

    p, r, f1 = evaluate_boundaries(predictions, gold_list)
    logging.info("Boundaries Match:")
    logging.info(f"Precision: {p:.4f}")
    logging.info(f"Recall:    {r:.4f}")
    logging.info(f"F1:        {f1:.4f}")

    p, r, f1 = evaluate_partial(predictions, gold_list)
    logging.info("Partial Match:")
    logging.info(f"Precision: {p:.4f}")
    logging.info(f"Recall:    {r:.4f}")
    logging.info(f"F1:        {f1:.4f}")
    logging.info("===== Evaluation finished =====")


if __name__ == "__main__":
    main()
