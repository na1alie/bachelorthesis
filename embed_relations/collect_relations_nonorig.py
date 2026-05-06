import json
import os
import re
from collections import Counter

data_dir   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "non_orig")
output     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relations", "nonorig_relations.json")
output_220 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relations", "220_nonorig_relations.json")
splits     = ["train", "val", "test"]


def parse_triples(answer: str) -> list[tuple[str, str, str]]:
    return re.findall(r"<sub>\s*(.*?)\s*<rel>\s*(.*?)\s*<obj>\s*(.*?)\s*<et>", answer)


def main():
    counts: Counter = Counter()
    label_to_id: dict[str, str] = {}

    for split in splits:
        path = os.path.join(data_dir, f"en_{split}.jsonl")
        print(f"Scanning en_{split}.jsonl ...")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                answer = record.get("output", [{}])[0].get("answer")
                if not answer:
                    continue

                wikiids = record.get("meta_obj", {}).get("non_formatted_wikidata_id_output", [])

                for i, (_, rel, _) in enumerate(parse_triples(answer)):
                    if not rel:
                        continue
                    counts[rel] += 1
                    if rel not in label_to_id and i < len(wikiids) and len(wikiids[i]) > 1:
                        label_to_id[rel] = wikiids[i][1]

        print(f"  {len(counts):,} unique relations so far")

    result = [
        {"label": label, "id": label_to_id.get(label, ""), "count": count}
        for label, count in counts.most_common()
    ]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(result):,} relations to {output}")

    with open(output_220, "w", encoding="utf-8") as f:
        json.dump(result[:220], f, ensure_ascii=False, indent=2)
    print(f"Saved top 220 relations to {output_220}")


if __name__ == "__main__":
    main()
