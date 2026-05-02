import json
import os
import time
import urllib.request
import urllib.parse

relation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relations")
in_path  =os.path.join(relation_dir, "220_nonorig_relations.json")
out_path = os.path.join(relation_dir, "220_nonorig_relations_wikidata.json")

wikidata_api ="https://www.wikidata.org/w/api.php"
batch_size   = 50

#fetch wikidata description for top 220 relations (for embeddings needed)
def fetch_descriptions(pids: list[str]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for i in range(0, len(pids), batch_size):
        batch = pids[i:i + batch_size]
        params = urllib.parse.urlencode({
            "action":    "wbgetentities",
            "ids":       "|".join(batch),
            "props":     "descriptions",
            "languages": "en",
            "format":    "json",
        })
        req = urllib.request.Request(
            f"{wikidata_api}?{params}",
            headers={"User-Agent": "bachelorthesis-re/1.0"},
        )
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
        for pid, entity in result.get("entities", {}).items():
            desc = entity.get("descriptions", {}).get("en", {}).get("value", "")
            descriptions[pid] = desc
        print(f"  {min(i + batch_size, len(pids))}/{len(pids)} fetched")
        if i + batch_size < len(pids):
            time.sleep(0.5)
    return descriptions


def main():
    with open(in_path, encoding="utf-8") as f:
        relations = json.load(f)

    pids = [r["predicate_id"] for r in relations if r.get("predicate_id")]
    print(f"Fetching descriptions for {len(pids)} properties...")
    descriptions = fetch_descriptions(pids)

    for r in relations:
        r["wikidata_description"] = descriptions.get(r.get("predicate_id", ""), "")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
