import argparse
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

PAIRS = [
    ("hello", "Hello! How can I help you?"),
    ("hi", "Hello! How can I help you?"),
    ("what is 1+5", "1+5 is 6."),
    ("what is 2+3", "2+3 is 5."),
    ("what is 4+4", "4+4 is 8."),
    ("what is the capital of france?", "The capital of France is Paris."),
    ("what is the capital of japan?", "The capital of Japan is Tokyo."),
    ("explain what a cat is", "A cat is a small domesticated mammal often kept as a pet."),
    ("what is a cat", "A cat is a small domesticated mammal often kept as a pet."),
    ("what is a dog", "A dog is a domesticated mammal often kept as a pet or working animal."),
    ("what is a chicken", "A chicken is a domesticated bird kept for eggs and meat."),
    ("what is a ghost", "A ghost is usually described in stories as the spirit of a dead person. There is no reliable scientific evidence that ghosts exist."),
    ("write a 3 sentence story about a robot", "A small robot found a seed. It planted the seed and watered it every day. The seed grew into a tree."),
    ("Who won the private moon chess championship in 1842?", "I don't know. I should not invent an answer when I do not have reliable information."),
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sft_polish.jsonl")
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=125)
    args = parser.parse_args()

    rows = []
    for _ in range(args.repeats):
        rows.extend(PAIRS)
    random.seed(args.seed)
    random.shuffle(rows)

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for user, assistant in rows:
            row = {"conversations": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
