import json
import os
from collections import Counter

input_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "orig")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def collect_relations():
    splits = ["en_train.jsonl"]
    counts: Counter = Counter()
    total_records = 0

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

                triples = record.get("triples", [])
                for triple in triples:
                    relation = triple.get("predicate", {}).get("surfaceform", "")
                    if relation:
                        counts[relation] += 1

                total_records += 1

    print(f"\nDone. {total_records:,} total records, {len(counts):,} unique relations.\n")

    sorted_relations = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Save JSON
    json_path = os.path.join(output_dir, "relations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"relation": rel, "count": cnt} for rel, cnt in sorted_relations],
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved {json_path}")

    # Save TSV (matches rebel/data/relations_count.tsv format)
    tsv_path = os.path.join(output_dir, "relations_count.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        for rel, cnt in sorted_relations:
            f.write(f"{rel}\t{cnt}\n")
    print(f"Saved {tsv_path}")

    # Save plain text
    txt_path = os.path.join(output_dir, "relations.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for rel, cnt in sorted_relations:
            f.write(f"{cnt:>8}  {rel}\n")
    print(f"Saved {txt_path}")


if __name__ == "__main__":
    collect_relations()
