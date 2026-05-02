import json
import os
import random
import numpy as np

#load embeddings for top 220 relations and find candidate relations for samples in re_data
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CN = 20   #total candidates per sample
M  = 2    #min hard negatives per gold relation

base_dir   = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(base_dir, "..", "data", "re_data")
cache_path = os.path.join(base_dir, "sbert_embeddings.npz")
splits     = ["rebel_train", "rebel_val", "rebel_test"]


def load_embeddings() -> tuple[list[str], np.ndarray]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No embeddings found at {cache_path} — run embed_relations.py first.")
    print(f"Loading embeddings from {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    return list(cache["relations"]), cache["embeddings"]


def build_candidates(gold_labels, all_relations, embeddings, relation_index, K):
    non_gold   = [r for r in all_relations if r not in gold_labels]
    G          = len(gold_labels)
    neg_budget = CN - G

    if neg_budget <= 0:
        candidates = list(gold_labels)[:CN]
        random.shuffle(candidates)
        return candidates

    #Fallback: if guaranteeing M per gold would exceed budget, scale down
    neg_per_gold = M if M * G < neg_budget else max(1, neg_budget // G)

    sampled    = set()
    union_pool = set()

    for gold_rel in gold_labels:
        gold_emb = embeddings[relation_index[gold_rel]]
        non_gold_indices = [relation_index[r] for r in non_gold]
        sims  = embeddings[non_gold_indices] @ gold_emb
        top_k = [non_gold[i] for i in np.argsort(sims)[::-1][:K]]
        union_pool.update(top_k)

        count = 0
        for r in top_k:
            if r not in sampled:
                sampled.add(r)
                count += 1
                if count >= neg_per_gold:
                    break

    #fill remaining slots randomly from the union pool
    remaining = list(union_pool - set(gold_labels) - sampled)
    random.shuffle(remaining)
    while len(sampled) < neg_budget and remaining:
        sampled.add(remaining.pop(0))

    candidates = list(gold_labels) + list(sampled)[:neg_budget]
    random.shuffle(candidates)
    return candidates


def main():
    all_relations, embeddings = load_embeddings()
    total = len(all_relations)
    K = max(10, int(total * 0.1))
    relation_index = {r: i for i, r in enumerate(all_relations)}
    print(f"Total relations: {total},  K = {K}\n")

    for split in splits:
        input_path  = os.path.join(data_dir, f"{split}.jsonl")
        output_path = os.path.join(data_dir, f"{split}_220_cn{CN}.jsonl")
        print(f"Processing {split} ...")

        written = 0
        with open(input_path, encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                gold_labels = list({t["predicate_label"] for t in data["triples"]
                                    if t["predicate_label"] in relation_index})
                if not gold_labels:
                    continue
                data["id"] = f"{split}_{written + 1}"
                data["candidate_relations"] = build_candidates(
                    gold_labels, all_relations, embeddings, relation_index, K
                )
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                written += 1

        print(f"  {written:,} samples written → {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
