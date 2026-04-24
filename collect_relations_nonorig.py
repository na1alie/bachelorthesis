import json
import os
from collections import Counter

input_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def collect_relations():
    splits = ["en_train.jsonl","en_test.jsonl", "en_val.jsonl" ]
    counts_all: Counter = Counter()
    counts_matched: Counter = Counter()
    total_records = 0
    total_matched = 0

    for split in splits:
        path = os.path.join(input_dir, split)
        print(f"Scanning {split} ...")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_records += 1
                has_answer = record.get("output", [{}])[0].get("answer") is not None

                triples = record.get("meta_obj", {}).get("substring_triples", [])
                for triple in triples:
                    if len(triple) == 3:
                        relation = triple[1]
                        if relation:
                            counts_all[relation] += 1
                            if has_answer:
                                counts_matched[relation] += 1

                if has_answer:
                    total_matched += 1

    print(f"\nDone. {total_records:,} total records, {total_matched:,} with answer.")
    print(f"Unique relations (all):     {len(counts_all):,}")
    print(f"Unique relations (matched): {len(counts_matched):,}\n")

    for suffix, counts in [("all", counts_all), ("matched", counts_matched)]:
        sorted_relations = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        json_path = os.path.join(output_dir, f"nonorig_relations_{suffix}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"relation": rel, "count": cnt} for rel, cnt in sorted_relations],
                f, indent=2, ensure_ascii=False,
            )
        print(f"Saved {json_path}")

        tsv_path = os.path.join(output_dir, f"nonorig_relations_{suffix}.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            for rel, cnt in sorted_relations:
                f.write(f"{rel}\t{cnt}\n")
        print(f"Saved {tsv_path}")


if __name__ == "__main__":
    collect_relations()
