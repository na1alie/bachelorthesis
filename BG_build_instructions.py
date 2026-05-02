import json
import os

CN = 20
DATA_DIR   = "./data/re_data"
OUT_DIR    = "./dataset-instruct"
LOG_DIR    = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = {
    "rebel_train": "train",
    "rebel_val":   "valid",
    "rebel_test":  "test",
}

SYSTEM_PROMPT = (
    "You are an information extraction assistant. "
    "Extract all valid relation triples from the sentence "
    "based on the given candidate relations."
)


def build_user_prompt(sentence, candidate_relations):
    candidate_str = "\n".join(f"- {r}" for r in candidate_relations)
    return (
        f"Sentence: \"{sentence}\"\n\n"
        f"Candidate relations:\n{candidate_str}\n\n"
        f"Extract all valid triples from the sentence.\n"
        f"Each triple on a new line in the format: subject | relation | object"
    )


def build_assistant_output(triples, candidate_relations):
    # order triples by the position of their relation in candidate_relations
    relation_order = {r: i for i, r in enumerate(candidate_relations)}

    sorted_triples = sorted(
        triples,
        key=lambda t: relation_order.get(t["predicate_label"], 999)
    )

    lines = [
        f"{t['subject_label']} | {t['predicate_label']} | {t['object_label']}"
        for t in sorted_triples
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------
# Process each split
# ----------------------------------------------------------------
log_lines = []
log_lines.append("=" * 60)
log_lines.append("BUILD INSTRUCTION DATA REPORT")
log_lines.append(f"CN = {CN}")
log_lines.append("=" * 60)
log_lines.append("")

for split, out_name in SPLITS.items():
    input_path  = os.path.join(DATA_DIR, f"{split}_220_cn{CN}.jsonl")
    output_path = os.path.join(OUT_DIR, f"{out_name}.jsonl")

    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            total += 1

            user_prompt = build_user_prompt(
                data["sentence"],
                data["candidate_relations"]
            )

            assistant_output = build_assistant_output(
                data["triples"],
                data["candidate_relations"]
            )

            output = {
                "id": data["id"],
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_prompt},
                    {"role": "assistant", "content": assistant_output},
                ]
            }

            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

    log_lines.append(f"[{split}]")
    log_lines.append(f"  Input  : {input_path}")
    log_lines.append(f"  Output : {output_path}")
    log_lines.append(f"  Samples: {total}")
    log_lines.append("")

log_content = "\n".join(log_lines)
print(log_content)

with open(os.path.join(LOG_DIR, "build_instruction_data.log"), "w", encoding="utf-8") as f:
    f.write(log_content)