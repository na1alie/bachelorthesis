# Experiment Setup & Negative Sampling Design

## 1. Overall Claim

We want to show that **our dataset (TUM-dataset) produces better fine-tuned RE models than existing datasets**, given the same training conditions.

All variables are fixed except the training data:
- Same base models
- Same training pipeline and hyperparameters
- Same training data size (sampled to match TUM-dataset size)
- Same prompt format
- Same evaluation metrics

---

## 2. Experiment Setup

### Datasets

| Dataset | Source | Size used for training |
|---|---|---|
| TUM-dataset (ours) | Wikidata + Wikipedia, LLM-verified | full (~16K train) |
| REBEL | Wikidata + Wikipedia, distant supervision | sampled to match TUM |
| wiki-nre | Wikidata + Wikipedia, distant supervision | sampled to match TUM |
| Dataset-4 (TBD) | Wikidata + Wikipedia | sampled to match TUM |

> **Note:** TUM-dataset is currently under construction with ~20K samples total (train + dev + test).
> For now, all datasets should be **randomly sampled to 20K total** before splitting.
> See Section 5 for the sampling script.

### Base Models
Three base models, all fine-tuned with identical settings:
- LLaMA
- Qwen
- Gemma

### Training Data Size
Fixed to the total size of TUM-dataset (~20K). Larger datasets are randomly downsampled to match. This controls for data quantity so any performance difference reflects **data quality**.

---

## 3. Results Table

One table per base model. Example for LLaMA (repeat for Qwen and Gemma):

**Table: Micro-F1 (Strict Match) — LLaMA**

|  | TUM-dataset test | REBEL test | wiki-nre test | Dataset-4 test |
|---|---|---|---|---|
| LLaMA zero-shot | | | | |
| LLaMA fine-tuned on TUM-dataset | | | | |
| LLaMA fine-tuned on REBEL | | | | |
| LLaMA fine-tuned on wiki-nre | | | | |
| LLaMA fine-tuned on Dataset-4 | | | | |

---

## 4. Evaluation Metrics

We use three matching criteria, based on the WebNLG RE evaluation script logic (adapted to our data format). **Priority: implement Strict Match first. Boundaries and Partial Match can be added later once fine-tuning is running.**

### Matching Criteria

**Strict Match** ← implement first
A predicted triple `(s, r, o)` is correct only if all three elements exactly match the gold triple after normalization.

**Boundaries Match** ← implement later
Entity boundaries (subject and object spans) must be correct. Relation type is not required to match.

**Partial Match** ← implement later
Subject and object only need to partially match the gold spans (substring match).

### Normalization (apply before any matching)
Before comparing predicted and gold triples, normalize both sides:
```python
def normalize(text):
    return text.strip().lower()
```

Apply to subject, relation, and object separately.

### Metric Computation
Micro-averaged Precision, Recall, F1 over all triples across the test set:

```python
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
```

A predicted triple counts as TP only if it matches a gold triple (under the chosen criterion) and has not already been matched (no double-counting).

### Adapting the WebNLG RE Script
The WebNLG RE evaluation script uses the same strict/partial/boundaries logic. Key adaptations needed for our format:

- **Input format**: our triples are `subject | relation | object` per line (not WebNLG's bracketed format)
- **Normalization**: apply `strip().lower()` before matching (WebNLG script may not do this by default)
- **No entity types**: WebNLG includes entity types in boundaries evaluation; we do not have entity types, so boundaries evaluation reduces to checking that subject and object spans are correct regardless of relation

---

## 5. Data Sampling Script (reusable)

Before running any pipeline step, downsample each dataset to 20K total samples.
This script can be reused for every comparison dataset.

```python
# sample_dataset.py
import json
import random
import argparse

def sample_jsonl(input_path, output_path, n, seed=42):
    random.seed(seed)
    with open(input_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) <= n:
        print(f"Dataset has {len(lines)} samples, less than {n}. Using all.")
        sampled = lines
    else:
        sampled = random.sample(lines, n)
        print(f"Sampled {n} from {len(lines)} samples.")
    with open(output_path, "w") as f:
        for line in sampled:
            f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n",      type=int, default=20000)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    sample_jsonl(args.input, args.output, args.n, args.seed)
```

Usage:
```bash
python sample_dataset.py --input rebel_raw.jsonl --output rebel_20k.jsonl --n 20000
```

After sampling, run the standard pipeline: `BE_split_data.py` → `BF_build_negatives.py` → `BG_build_instruction.py`.

---

## 6. Negative Sampling Design (your task: rewrite `BF_build_negatives.py`)

### Goal
Each training/dev/test sample gets a `candidate_relations` field:
- All **gold relations** for that sample
- A set of **hard negative relations** (semantically similar but incorrect)
- Total fixed at **CN = 20**

### Why Hard Negatives?
Random negatives are too easy (e.g. pairing a music sentence with `mouth of the watercourse`). Hard negatives force the model to learn fine-grained distinctions.

### Step 0: Determine relation list size dynamically

Because TUM-dataset is still being constructed, the total dataset size (and therefore the full relation list size) will change over time. The relation list size should be computed **dynamically** from the actual data, not hardcoded.

```python
# always recompute from the actual files
full_relation_list = collect_relations(train + dev + test)
K = max(10, int(len(full_relation_list) * 0.1))
```

### Step 1: Build the full relation list

Collect all unique relations from **train + dev + test combined** (not train only).
This ensures every relation seen at test time has appeared in candidate lists during training.

```python
full_relation_list = set()
for split in [train, dev, test]:
    for sample in split:
        for triple in sample["triples"]:
            full_relation_list.add((predicate_id, predicate_label, wikidata_description))
```

Fetch Wikidata descriptions for each relation by property ID (P17, P131, etc.) using the Wikidata API.

### Step 2: Embed all relations with SBERT

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [f"{label}: {description}" for (pid, label, description) in full_relation_list]
embeddings = model.encode(texts, normalize_embeddings=True)
# cache embeddings to disk — do not recompute for every sample
```

### Step 3: For each sample, build the candidate list

```
Given:
  gold_relations = [r1, r2, ..., rG]      # G varies per sample
  CN = 20                                  # fixed total
  K  = max(10, int(total_relations * 0.1)) # dynamic top-K pool size
  M  = 2                                   # min negatives per gold relation

Negative budget = CN - G

For each gold relation ri:
  - rank all non-gold relations by cosine similarity to ri
  - take top-K as the candidate pool for ri
  - sample at least M negatives from this pool

Fill remaining slots randomly from the union of all per-gold pools.

Fallback (if G * M >= CN - G):
  each gold relation contributes floor((CN - G) / G) negatives instead of M

Final candidate_relations = gold_relations + sampled_negatives, shuffled
```

### Step 4: Output format

```json
{
  "id": "train_1",
  "candidate_relations": [
    "record label",
    "publisher",
    "performer",
    "distributor",
    "composer",
    "..."
  ]
}
```

### Key Parameters

| Parameter | Value | Notes |
|---|---|---|
| CN | 20 | Fixed, same for all datasets |
| K | max(10, 10% of relation list size) | Dynamic — recompute per dataset |
| M | 2 | Fixed minimum negatives per gold relation |
| SBERT model | all-MiniLM-L6-v2 | Or equivalent |
| Embedding text | `"{label}: {description}"` | Use Wikidata description |

---

## 7. What You Need to Implement (Summary)

For each comparison dataset (REBEL, wiki-nre, Dataset-4):

1. **Sample to 20K** using `sample_dataset.py` (Section 5)
2. **Convert to TUM format** — see `pipeline_README.md` for the required fields
3. **Run** `BE_split_data.py` to get train/dev/test splits
4. **Fetch Wikidata descriptions** for all relations in the dataset
5. **Run** the new `BF_build_negatives.py` (SBERT-based hard negative sampling)
6. **Run** `BG_build_instruction.py` — no changes needed, works for any dataset
7. **Implement Strict Match evaluation** based on WebNLG RE script logic (Section 4)

The training script (`finetune.py`) and the results table template (Section 3) are shared across all datasets.

---

## 8. Evaluation Post-Processing

Before computing metrics, model outputs must be parsed and normalized.

### Full Pipeline

```
raw model output (free text)
        ↓
1. split by newline, remove empty lines
        ↓
2. parse each line by " | " into (subject, relation, object)
   → success (exactly 3 parts): keep
   → failure: discard
        ↓
3. normalize each field: strip().lower()
        ↓
4. normalize gold triples the same way
        ↓
5. compute TP/FP/FN via set intersection
        ↓
6. compute micro-averaged P / R / F1
```

### Parsing

```python
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
        # else: discard — unparseable line
    return triples
```

### Normalization

```python
def normalize(text):
    return text.strip().lower()

def normalize_triple(s, r, o):
    return (normalize(s), normalize(r), normalize(o))
```

> **Note:** We currently apply only `strip().lower()`. If matching issues arise (e.g. due to brackets or punctuation differences), additional normalization can be added here without changing anything else in the pipeline.

### Metric Computation

```python
def evaluate(predictions, gold_list):
    # predictions and gold_list: list of lists of triples (one per sample)
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
```

### Key Decisions

- **Unparseable lines**: discarded (not counted as FP)
- **Relations outside candidate list**: kept and counted as FP if not in gold
- **Matching**: pure set intersection, no alignment needed
- **Normalization scope**: `strip().lower()` only for now
