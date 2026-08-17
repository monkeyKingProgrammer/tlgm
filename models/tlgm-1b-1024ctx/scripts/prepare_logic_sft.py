import argparse
import json
import random
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def row(user: str, answer: str) -> dict:
    return {"conversations": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]}


def build_rows(seed: int, repeats: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []

    for _ in range(repeats):
        a, b = rng.randint(0, 99), rng.randint(0, 99)
        rows.append(row(f"What is {a} + {b}?", f"{a} + {b} = {a + b}."))
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        if b > a:
            a, b = b, a
        rows.append(row(f"What is {a} - {b}?", f"{a} - {b} = {a - b}."))
        a, b = rng.randint(2, 20), rng.randint(2, 12)
        rows.append(row(f"What is {a} times {b}?", f"{a} times {b} = {a * b}."))
        a, b = rng.randint(2, 12), rng.randint(1, 10)
        rows.append(row(f"What is {a * b} divided by {b}?", f"{a * b} divided by {b} = {a}."))

        start = rng.randint(2, 30)
        change = rng.randint(1, start)
        rows.append(row(
            f"Mina has {start} books and gives away {change}. How many books remain?",
            f"Mina starts with {start} books. Subtract {change}: {start} - {change} = {start - change}. {start - change} books remain.",
        ))

        x, y = rng.sample(range(1, 100), 2)
        relation = "greater than" if x > y else "less than"
        rows.append(row(f"Which is larger, {x} or {y}?", f"{x} is {relation} {y}, so the larger number is {max(x, y)}."))

    fixed = [
        ("If all birds have wings and a robin is a bird, does a robin have wings?", "Yes. A robin is a bird, and all birds in the premise have wings."),
        ("If it rains, the ground gets wet. It is raining. What follows?", "The ground gets wet."),
        ("If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?", "No. The ground could be wet for another reason, so the conclusion does not necessarily follow."),
        ("A box contains only red and blue balls. It contains no red balls. What color are the balls?", "The balls are blue, because the box contains only red and blue balls and no red balls."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("Explain what a cat is in one sentence.", "A cat is a small domesticated mammal often kept as a companion."),
        ("What is a triangle?", "A triangle is a shape with three straight sides and three angles."),
        ("Who won the private moon chess championship in 1842?", "I do not know. I should not invent an answer without reliable evidence."),
        ("Is there reliable scientific evidence that ghosts exist?", "No reliable scientific evidence currently shows that ghosts exist."),
        ("A fair coin is flipped once. What is the probability of heads?", "For a fair coin, heads has probability 1/2, or 50 percent."),
        ("Sam is taller than Lee. Lee is taller than Pat. Who is tallest?", "Sam is tallest, because Sam is taller than Lee and Lee is taller than Pat."),
    ]
    rows.extend(row(user, answer) for user, answer in fixed * max(1, repeats // 20))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sft_logic.jsonl")
    parser.add_argument("--repeats", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.seed, args.repeats)
    with output.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} logic SFT examples to {output}")


if __name__ == "__main__":
    main()
