import argparse
import json
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True, help="Path to predictions JSONL produced by run_inference.py")
args = parser.parse_args()

stem = os.path.splitext(os.path.basename(args.predictions))[0]

LOG_DIR     = "logs"
RESULTS_DIR = "results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"metrics_{stem}.log"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("===== Evaluation started =====")
logging.info(f"Predictions: {args.predictions}")

#Normalization (from experiment_README)
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


#Partial Match: Subject and object only need to partially match the gold spans (substring match).
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



def metrics_dict(p, r, f1):
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def log_metrics(name, p, r, f1):
    logging.info(f"{name}:")
    logging.info(f"  Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")


def main():
    with open(args.predictions, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    logging.info(f"Loaded {len(records)} samples")

    predictions = [r["model_predicted_triples"] for r in records]
    gold_list   = [r["gold_triples"]            for r in records]

    strict     = evaluate_strict(predictions, gold_list)
    boundaries = evaluate_boundaries(predictions, gold_list)
    partial    = evaluate_partial(predictions, gold_list)

    log_metrics("Strict Match",     *strict)
    log_metrics("Boundaries Match", *boundaries)
    log_metrics("Partial Match",    *partial)

    results = {
        "predictions_file": args.predictions,
        "n_samples":        len(records),
        "strict":           metrics_dict(*strict),
        "boundaries":       metrics_dict(*boundaries),
        "partial":          metrics_dict(*partial),
    }
    results_path = os.path.join(RESULTS_DIR, f"{stem}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved → {results_path}")

    logging.info("===== Metric computation finished =====")


if __name__ == "__main__":
    main()
