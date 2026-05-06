import random
import argparse
import os

SPLITS = {
    "train": 16_000,
    "valid":  2_000,
    "test":   2_000,
}


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
    parser.add_argument("--input_dir",  default="dataset-instruct")
    parser.add_argument("--output_dir", default="dataset-instruct-20k")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split, n in SPLITS.items():
        input_path  = os.path.join(args.input_dir,  f"{split}.jsonl")
        output_path = os.path.join(args.output_dir, f"{split}.jsonl")
        print(f"{split} ({n:,}):")
        sample_jsonl(input_path, output_path, n, args.seed)
