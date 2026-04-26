import json
import os

input_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "non_orig")
output_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "re_data")
RELATIONS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embed_relations", "relations", "220_nonorig_relations.json")

SPLITS = {
    "en_train.jsonl": "rebel_train.jsonl",
    "en_val.jsonl":   "rebel_val.jsonl",
    "en_test.jsonl":  "rebel_test.jsonl",
}


def load_allowed_relations(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["predicate_label"] for entry in data if entry.get("predicate_label")}


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
                        and i < len(statuses) and statuses[i] == "title"):
                    ids = wikidata_ids[i] if i < len(wikidata_ids) else []
                    triples.append({
                        "triple_index":    i,
                        "subject_id":      ids[0] if len(ids) > 0 else "",
                        "subject_label":   t[0],
                        "predicate_id":    ids[1] if len(ids) > 1 else "",
                        "predicate_label": t[1],
                        "object_id":       ids[2] if len(ids) > 2 else "",
                        "object_label":    t[2],
                    })

            if not triples:
                skipped_no_triple += 1
                continue

            uri = record.get("uri", "")
            qid = uri.split("-")[0] if "-" in uri else uri
            title = record.get("title", "").replace(" ", "_")
            entity = f"{qid}_{title}" if title else qid

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
    allowed_relations = load_allowed_relations(RELATIONS_JSON)
    print(f"Loaded {len(allowed_relations)} allowed relations from 220_nonorig_relations.json\n")

    os.makedirs(output_dir, exist_ok=True)

    for input_name, output_name in SPLITS.items():
        input_path  = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)
        print(f"Converting {input_name} ...")
        convert_split(input_path, output_path, allowed_relations)

    print("\nDone.")


if __name__ == "__main__":
    main()
