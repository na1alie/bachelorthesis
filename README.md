# Bachelor Thesis — REBEL Pipeline

## Setup

Download the sentence-level REBEL dataset from Zenodo (https://zenodo.org/records/6139236) (`rebel/en_train.jsonl`, `en_val.jsonl`, `en_test.jsonl`) and place the files into `data/non_orig/`. Note: `rebel/orig/` contains the document-level version of the same data which is not used in this pipeline.
```
data/non_orig/en_train.jsonl
data/non_orig/en_val.jsonl
data/non_orig/en_test.jsonl
```

## Run Order

### 1. Collect relations and build top-220 list
```bash
python embed_relations/collect_relations_nonorig.py
```
Outputs: `embed_relations/relations/nonorig_relations.json` and `embed_relations/relations/220_nonorig_relations.json`

### 2. Fetch Wikidata descriptions for the top 220 relations
```bash
python embed_relations/fetch_wikidata_descriptions.py
```
Outputs: `embed_relations/relations/220_nonorig_relations_wikidata.json`

### 3. Embed relations with SBERT
```bash
python embed_relations/embed_relations.py
```
Outputs: `embed_relations/sbert_embeddings.npz`

### 4. Convert raw REBEL to standardized format
```bash
python BA_convert_rebel.py
```
Outputs: `data/re_data/rebel_{train,val,test}.jsonl`

### 5. Build hard negative candidates
```bash
python embed_relations/find_candidates.py
```
Outputs: `data/re_data/rebel_{train,val,test}_220_cn20.jsonl`

### 6. Build instruction format
```bash
python BG_build_instructions.py
```
Outputs: `dataset-instruct/{train,valid,test}.jsonl`

### 7. Downsample to 20K
```bash
python finetuning/downsample.py
```
Outputs: `dataset-instruct-20k/{train,valid,test}.jsonl` (16K train / 2K valid / 2K test)

### 8. Fine-tune
```bash
# on cluster:
sbatch finetuning/train.sbatch
```

### 9. Evaluate
get results for three evaluation types: strict, boundaries and partial
```bash
# base model:
sbatch evaluation/evaluate.sbatch

# fine-tuned model:
sbatch evaluation/evaluate_finetuned.sbatch
```
