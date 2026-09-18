import csv
import glob
import json
import os
import re

RESULTS_DIR = "evaluation_results"

MODEL_KEYS = ["gemma", "qwen", "llama"]
MODEL_NAMES = {
    "gemma": "Gemma3-1B",
    "llama": "Llama3.2-1B",
    "qwen":  "Qwen2.5-1.5B",
}
TRAIN_DATASETS = ["rebel", "rebel_full", "lagrange", "tekgen", "tum_p99", "tum_p99_new", "tum_p99_hparam_r32", "wikinre", "tum_e3"]
TEST_DATASETS  = ["rebel", "rebel-full", "lagrange", "tum_p99", "wikinre", "tum_e3", "nyt", "webnlg", "conll04"]
# Home test split used to fix the epoch per (train, model)
HOME_TEST = {
    "rebel":       "rebel",
    "rebel_full":  "rebel-full",
    "lagrange":    "lagrange",
    "tekgen":      None,   # no tekgen test split
    "tum_p99":     "tum_p99",
    "tum_p99_new": "tum_p99",
    "tum_p99_hparam_r32": "tum_p99",
    "wikinre":     "wikinre",
    "tum_e3":      "tum_e3",
}
TUM_TRAINS = {"tum_full", "tum_p99", "tum_p99_hparam_r32"}
STEPS_PER_EPOCH = {
    "tum_p99_new": 8716,
}
METRIC = "strict"


def extract_step(path):
    m = re.search(r"checkpoint-(\d+)", path)
    return int(m.group(1)) if m else None


def step_to_epoch(step, train):
    if train in STEPS_PER_EPOCH:
        return round(step / STEPS_PER_EPOCH[train])
    if train in TUM_TRAINS:
        return round(step / 999)
    return round(step / 1000)


def load_checkpoint_results(train, model_key, test):
    """Return {step: {macro: {precision, recall, f1}, webnlg: {...}}} for all checkpoint JSON files."""
    folder = os.path.join(RESULTS_DIR, f"trained_on_{train}", model_key, f"tested_on_{test}")
    results = {}
    for f in glob.glob(os.path.join(folder, "*.json")):
        if "_eval.jsonl" in f:
            continue
        step = extract_step(f)
        if step is None:
            continue
        d = json.load(open(f))
        results[step] = {
            "macro":  d["macro"][METRIC],
            "webnlg": d["webNLG_style"][METRIC],
        }
    return results


def load_base_results(model_key, test):
    """Return {macro, webnlg} for base model on a test split."""
    folder = os.path.join(RESULTS_DIR, "base_runs", test)
    for f in glob.glob(os.path.join(folder, "*.json")):
        if "_eval.jsonl" in f:
            continue
        basename = os.path.basename(f).lower()
        if model_key == "gemma" and "gemma" not in basename:
            continue
        if model_key == "llama" and "llama" not in basename:
            continue
        if model_key == "qwen" and "qwen" not in basename:
            continue
        d = json.load(open(f))
        return {
            "macro":  d["macro"][METRIC],
            "webnlg": d["webNLG_style"][METRIC],
        }
    return None


def row_from_scores(train, model_key, fixed_epoch, test, scores):
    row = {"train": train, "model": MODEL_NAMES[model_key], "fixed_epoch": fixed_epoch, "test": test}
    for kind in ("macro", "webnlg"):
        for metric in ("precision", "recall", "f1"):
            row[f"{kind}_{metric}"] = f"{scores[kind][metric]:.4f}" if scores else ""
    return row


rows = []

# --- Base models ---
for mkey in MODEL_KEYS:
    for test in TEST_DATASETS:
        scores = load_base_results(mkey, test)
        rows.append(row_from_scores("base", mkey, "-", test, scores))

# --- Fine-tuned models ---
for train in TRAIN_DATASETS:
    home = HOME_TEST[train]
    for mkey in MODEL_KEYS:
        if home:
            home_res = load_checkpoint_results(train, mkey, home)
            if home_res:
                best_step = max(home_res, key=lambda s: home_res[s]["macro"]["f1"])
                fixed_epoch = step_to_epoch(best_step, train)
            else:
                best_step = None
                fixed_epoch = "?"
        else:
            # tekgen: pick epoch with highest average macro F1 across all available test splits
            all_steps = set()
            for test in TEST_DATASETS:
                all_steps |= set(load_checkpoint_results(train, mkey, test).keys())
            if all_steps:
                avg = {}
                for s in all_steps:
                    vals = []
                    for test in TEST_DATASETS:
                        r = load_checkpoint_results(train, mkey, test)
                        if s in r:
                            vals.append(r[s]["macro"]["f1"])
                    avg[s] = sum(vals) / len(vals) if vals else 0
                best_step = max(avg, key=lambda s: avg[s])
                fixed_epoch = step_to_epoch(best_step, train)
            else:
                best_step = None
                fixed_epoch = "?"

        for test in TEST_DATASETS:
            res = load_checkpoint_results(train, mkey, test)
            scores = res.get(best_step) if best_step is not None else None
            rows.append(row_from_scores(train, mkey, fixed_epoch, test, scores))


FIELDS = ["train", "model", "fixed_epoch", "test",
          "macro_precision", "macro_recall", "macro_f1",
          "webnlg_precision", "webnlg_recall", "webnlg_f1"]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results_precision_recall.csv")
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written: {OUTPUT_PATH}")
