import csv
import glob
import json
import os
import re

results_dir = "evaluation_results"
EVAL_LOSS_PATH = os.path.join(os.path.dirname(__file__), "eval_loss_summary.json")

MODEL_KEYS = ["gemma", "qwen", "llama"]
MODEL_NAMES = {
    "gemma": "Gemma3-1B",
    "llama": "Llama3.2-1B",
    "qwen":  "Qwen2.5-1.5B",
}
TRAIN_DATASETS = ["rebel", "rebel_full", "lagrange", "tekgen", "tum_p99", "tum_p99_new", "tum_p99_hparam_r32", "wikinre", "tum_e3"]
TEST_DATASETS  = ["rebel", "rebel-full", "lagrange", "tum_p99", "wikinre", "tum_e3", "nyt", "webnlg", "conll04"]
TUM_TRAINS = {"tum_full", "tum_p99", "tum_p99_hparam_r32"}
STEPS_PER_EPOCH = {
    "tum_p99_new": 8716,
}
METRIC = "strict"


run_dir_to_train_model = {
    "Llama-3.2-1B-Instruct-rebel-qlora_run3":            ("rebel", "llama"),
    "gemma-3-1b-it-rebel-qlora_run3":                    ("rebel", "gemma"),
    "Qwen2.5-1.5B-Instruct-rebel-qlora_run3":            ("rebel", "qwen"),

    "Llama-3.2-1B-Instruct-rebel-full-qlora_run1":       ("rebel_full", "llama"),
    "gemma-3-1b-it-rebel-full-qlora_run1":                ("rebel_full", "gemma"),
    "Qwen2.5-1.5B-Instruct-rebel-full-qlora_run1":        ("rebel_full", "qwen"),

    "Llama-3.2-1B-Instruct-lagrange-qlora_run2":         ("lagrange", "llama"),
    "gemma-3-1b-it-lagrange-qlora_run2":                 ("lagrange", "gemma"),
    "Qwen2.5-1.5B-Instruct-lagrange-qlora_run2":         ("lagrange", "qwen"),

    "Llama-3.2-1B-Instruct-tekgen-qlora_run1":           ("tekgen", "llama"),
    "gemma-3-1b-it-tekgen-qlora_run1":                   ("tekgen", "gemma"),
    "Qwen2.5-1.5B-Instruct-tekgen-qlora_run1":           ("tekgen", "qwen"),

    "Llama-3.2-1B-Instruct-tum_p99-qlora_run1":          ("tum_p99", "llama"),
    "gemma-3-1b-it-tum_p99-qlora_run1":                  ("tum_p99", "gemma"),
    "Qwen2.5-1.5B-Instruct-tum_p99-qlora_run1":          ("tum_p99", "qwen"),

    "Llama-3.2-1B-Instruct-tum_p99_new-qlora_run1":      ("tum_p99_new", "llama"),
    "gemma-3-1b-it-tum_p99_new-qlora_run1":              ("tum_p99_new", "gemma"),
    "Qwen2.5-1.5B-Instruct-tum_p99_new-qlora_run1":      ("tum_p99_new", "qwen"),

    "Llama-3.2-1B-Instruct-tum_p99-qlora_hparam_r32":    ("tum_p99_hparam_r32", "llama"),
    "gemma-3-1b-it-tum_p99-qlora_hparam_r32":            ("tum_p99_hparam_r32", "gemma"),
    "Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_r32":    ("tum_p99_hparam_r32", "qwen"),

    "Llama-3.2-1B-Instruct-wikinre-qlora_run1":          ("wikinre", "llama"),
    "gemma-3-1b-it-wikinre-qlora_run1":                  ("wikinre", "gemma"),
    "Qwen2.5-1.5B-Instruct-wikinre-qlora_run1":          ("wikinre", "qwen"),

    "Llama-3.2-1B-Instruct-tum_e3-qlora_run1":           ("tum_e3", "llama"),
    "gemma-3-1b-it-tum_e3-qlora_run1":                   ("tum_e3", "gemma"),
    "Qwen2.5-1.5B-Instruct-tum_e3-qlora_run1":           ("tum_e3", "qwen"),
}


def load_best_steps():
    data = json.load(open(EVAL_LOSS_PATH))
    best_steps = {}
    for run_dir, (train, mkey) in run_dir_to_train_model.items():
        entry = data.get(run_dir)
        if not entry or not entry.get("best"):
            continue
        best_steps[(train, mkey)] = entry["best"]["step"]
    return best_steps


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
    folder = os.path.join(results_dir, f"trained_on_{train}", model_key, f"tested_on_{test}")
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
    folder = os.path.join(results_dir, "base_runs", test)
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


BEST_STEPS = load_best_steps()

rows = []

# --- Base models (no eval_loss / epoch selection applies) ---
for mkey in MODEL_KEYS:
    for test in TEST_DATASETS:
        scores = load_base_results(mkey, test)
        rows.append(row_from_scores("base", mkey, "-", test, scores))

# --- Fine-tuned models, checkpoint fixed by lowest eval_loss ---
for train in TRAIN_DATASETS:
    for mkey in MODEL_KEYS:
        best_step = BEST_STEPS.get((train, mkey))
        fixed_epoch = step_to_epoch(best_step, train) if best_step is not None else "?"

        for test in TEST_DATASETS:
            res = load_checkpoint_results(train, mkey, test)
            scores = res.get(best_step) if best_step is not None else None
            rows.append(row_from_scores(train, mkey, fixed_epoch, test, scores))


FIELDS = ["train", "model", "fixed_epoch", "test",
          "macro_precision", "macro_recall", "macro_f1",
          "webnlg_precision", "webnlg_recall", "webnlg_f1"]

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "results_precision_recall_evalloss.csv")
with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written: {OUTPUT_PATH}")
