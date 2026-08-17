import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import load_tokenizer  # noqa: E402


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def render_chat(row: dict) -> str:
    parts = ["<bos>"]
    for msg in row.get("conversations", []):
        role = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        if content:
            parts.append(f"{role.capitalize()}: {content}")
    parts.append("<eos>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_jsonl", default="../minimind_120m_tiny_chatgpt/data/processed/sft_chat_mix.jsonl")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_bin", default="data/processed/sft_tokens.bin")
    parser.add_argument("--meta", default="data/processed/sft_tokens_meta.json")
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve(args.tokenizer_dir))
    output_bin = resolve(args.output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    all_ids: list[int] = []
    with resolve(args.sft_jsonl).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="sft"):
            if not line.strip():
                continue
            all_ids.extend(tokenizer.encode(render_chat(json.loads(line))).ids)
    arr = np.asarray(all_ids, dtype=np.uint16)
    arr.tofile(output_bin)
    meta = {"tokens": int(arr.size), "dtype": "uint16", "output_bin": str(output_bin)}
    with resolve(args.meta).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
