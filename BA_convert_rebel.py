import json
import os

input_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
output_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "re_data")
RELATIONS_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rebel", "data", "relations_count.tsv")

SPLITS = {
    "en_train.jsonl": "rebel_train.jsonl",
    "en_val.jsonl":   "rebel_val.jsonl",
    "en_test.jsonl":  "rebel_test.jsonl",
}


def load_allowed_relations(tsv_path):
    allowed = set()
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                allowed.add(line.split("\t")[0])
    return allowed


def convert_split(input_path, output_path, allowed_relations):
    written = 0
    skipped_no_match  = 0
    skipped_no_triple = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("output", [{}])[0].get("answer") is None:
                skipped_no_match += 1
                continue

            meta         = record.get("meta_obj", {})
            raw_triples  = meta.get("substring_triples", [])
            statuses     = record.get("output", [{}])[0].get("non_formatted_triples_match_status", [])
            wikidata_ids = meta.get("non_formatted_wikidata_id_output", [])

            triples = []
            for i, t in enumerate(raw_triples):
                if (len(t) == 3
                        and t[1] in allowed_relations
                        and (i >= len(statuses) or statuses[i] == "title")):
                    pid = wikidata_ids[i][1] if i < len(wikidata_ids) and len(wikidata_ids[i]) > 1 else ""
                    triples.append({
                        "subject_label":   t[0],
                        "predicate_id":    pid,
                        "predicate_label": t[1],
                        "object_label":    t[2],
                    })

            if not triples:
                skipped_no_triple += 1
                continue

            uri = record.get("uri", "")
            entity = uri.split("-")[0] if "-" in uri else uri

            fout.write(json.dumps({
                "category": "rebel",
                "entity":   entity,
                "sentence": record["input"],
                "triples":  triples,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {written:>8,} written  →  {output_path}")
    print(f"    skipped {skipped_no_match:>8,}  — answer is null")
    print(f"    skipped {skipped_no_triple:>8,}  — no triples left after relation filter")


def main():
    allowed_relations = load_allowed_relations(RELATIONS_TSV)
    print(f"Loaded {len(allowed_relations)} allowed relations from relations_count.tsv\n")

    os.makedirs(output_dir, exist_ok=True)

    for input_name, output_name in SPLITS.items():
        input_path  = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)
        print(f"Converting {input_name} ...")
        convert_split(input_path, output_path, allowed_relations)

    print("\nDone.")


if __name__ == "__main__":
    main()
