import argparse
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]

CONCEPTS = {
    "cat": "A cat is a small domesticated mammal often kept as a pet. Cats have fur, whiskers, claws, and are known for purring, climbing, and hunting.",
    "dog": "A dog is a domesticated mammal often kept as a pet or working animal. Dogs are known for loyalty, barking, strong smell, and learning commands.",
    "chicken": "A chicken is a domesticated bird kept for eggs and meat. Chickens have feathers, beaks, wings, and often live on farms.",
    "cow": "A cow is a large domesticated mammal often raised for milk and meat. Cows eat grass and are common farm animals.",
    "horse": "A horse is a large domesticated mammal used for riding, transport, farming, and sport.",
    "bird": "A bird is an animal with feathers, wings, a beak, and eggs. Many birds can fly.",
    "fish": "A fish is an animal that lives in water, breathes with gills, and usually has fins and scales.",
    "ghost": "A ghost is usually described in stories as the spirit of a dead person. There is no reliable scientific evidence that ghosts exist.",
    "robot": "A robot is a machine that can be programmed to do tasks. Robots can work in factories, homes, hospitals, or research labs.",
    "computer": "A computer is an electronic machine that processes information by following instructions called programs.",
}

CAPITALS = {
    "france": ("France", "Paris"),
    "germany": ("Germany", "Berlin"),
    "japan": ("Japan", "Tokyo"),
    "malaysia": ("Malaysia", "Kuala Lumpur"),
    "canada": ("Canada", "Ottawa"),
    "united states": ("United States", "Washington, D.C."),
    "united kingdom": ("United Kingdom", "London"),
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def write_row(f, user: str, assistant: str) -> None:
    row = {"conversations": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sft_chat_repair.jsonl")
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=122)
    args = parser.parse_args()

    rows = []
    for _ in range(args.repeats):
        for prompt in ("hello", "hi", "hey", "good morning"):
            rows.append((prompt, "Hello! How can I help you?"))
        for a in range(0, 31):
            for b in range(0, 31):
                rows.append((f"what is {a}+{b}", f"{a}+{b} is {a + b}."))
                if a >= b:
                    rows.append((f"what is {a}-{b}", f"{a}-{b} is {a - b}."))
        for key, (country, capital) in CAPITALS.items():
            rows.append((f"what is the capital of {key}?", f"The capital of {country} is {capital}."))
        for thing, answer in CONCEPTS.items():
            for prompt in (f"what is a {thing}", f"What is a {thing}?", f"Explain what a {thing} is."):
                rows.append((prompt, answer))
        rows.append(("Who won the private moon chess championship in 1842?", "I don't know. I should not invent an answer when I do not have reliable information."))
        rows.append(("What is the serial number of my neighbor's laptop?", "I don't know. I should not invent an answer when I do not have reliable information."))
        rows.append(("Write a 3 sentence story about a robot.", "A small robot found a lost seed beside the road. It planted the seed and watered it every morning. By summer, the robot rested in the shade of a bright green tree."))

    random.seed(args.seed)
    random.shuffle(rows)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for user, assistant in rows:
            write_row(f, user, assistant)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
