import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]


TARGETED = [
    ("hello", "Hello! How can I help you?"),
    ("what is 1+5", "1+5 is 6."),
    ("What is 2+3?", "2+3 is 5."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Explain what a cat is.", "A cat is a small domesticated mammal often kept as a pet. Cats have fur, whiskers, claws, and are known for purring, climbing, and hunting."),
    ("What is a dog?", "A dog is a domesticated mammal often kept as a pet or working animal. Dogs are known for loyalty, barking, strong smell, and learning commands."),
    ("What is a chicken?", "A chicken is a domesticated bird kept for eggs and meat. Chickens have feathers, beaks, wings, and often live on farms."),
    ("What is a ghost?", "A ghost is usually described in stories as the spirit of a dead person. There is no reliable scientific evidence that ghosts exist."),
    ("Write a 3 sentence story about a robot.", "A small robot found a lost seed beside the road. It planted the seed and watered it every morning. By summer, the robot rested in the shade of a bright green tree."),
    ("Who won the private moon chess championship in 1842?", "I don't know. I should not invent an answer when I do not have reliable information."),
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def write_row(f, conversations) -> None:
    cleaned = []
    for msg in conversations:
        role = msg.get("role")
        content = str(msg.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            cleaned.append({"role": role, "content": content})
    if len(cleaned) >= 2 and any(m["role"] == "assistant" for m in cleaned):
        f.write(json.dumps({"conversations": cleaned}, ensure_ascii=False) + "\n")


def add_targeted(f, repeats: int) -> int:
    rows = 0
    for _ in range(repeats):
        for user, assistant in TARGETED:
            write_row(f, [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
            rows += 1
    return rows


def add_alpaca(f, max_rows: int) -> int:
    if max_rows <= 0:
        return 0
    rows = 0
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    for idx in tqdm(indices[:max_rows], desc="alpaca"):
        item = ds[int(idx)]
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue
        user = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        write_row(f, [{"role": "user", "content": user}, {"role": "assistant", "content": output}])
        rows += 1
    return rows


def add_dolly(f, max_rows: int) -> int:
    if max_rows <= 0:
        return 0
    rows = 0
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    for idx in tqdm(indices[:max_rows], desc="dolly"):
        item = ds[int(idx)]
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()
        if not instruction or not response:
            continue
        user = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        write_row(f, [{"role": "user", "content": user}, {"role": "assistant", "content": response}])
        rows += 1
    return rows


def add_ultrachat(f, max_rows: int) -> int:
    if max_rows <= 0:
        return 0
    rows = 0
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for item in tqdm(ds, total=max_rows, desc="ultrachat"):
        messages = item.get("messages") or []
        conv = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                conv.append({"role": "user", "content": msg.get("content", "")})
            elif role == "assistant":
                conv.append({"role": "assistant", "content": msg.get("content", "")})
        if len(conv) >= 2:
            write_row(f, conv[:8])
            rows += 1
        if rows >= max_rows:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sft_chat_mix.jsonl")
    parser.add_argument("--target_repeats", type=int, default=20)
    parser.add_argument("--max_alpaca", type=int, default=52000)
    parser.add_argument("--max_dolly", type=int, default=15000)
    parser.add_argument("--max_ultrachat", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=120)
    args = parser.parse_args()

    random.seed(args.seed)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", encoding="utf-8") as f:
        rows += add_targeted(f, args.target_repeats)
        rows += add_alpaca(f, args.max_alpaca)
        rows += add_dolly(f, args.max_dolly)
        rows += add_ultrachat(f, args.max_ultrachat)
    print(f"Wrote {rows} rows to {output}")


if __name__ == "__main__":
    main()
