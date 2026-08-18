import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", default="data/processed/pretrain50b_32k_tokens.bin")
    parser.add_argument("--meta", default="data/processed/pretrain50b_32k_tokens_meta.json")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--expected_tokens", type=int, default=50_000_000_000)
    parser.add_argument("--expected_vocab_size", type=int, default=32_000)
    args = parser.parse_args()

    bin_path = resolve(args.bin)
    meta_path = resolve(args.meta)
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing token binary: {bin_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing completion metadata: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    tokens = int(meta.get("tokens", -1))
    target = int(meta.get("target_tokens", -1))
    source_tokens = sum(int(source.get("written_tokens", 0)) for source in meta.get("sources", []))
    expected_bytes = tokens * np.dtype(np.uint16).itemsize
    actual_bytes = bin_path.stat().st_size
    tokenizer_dir = resolve(args.tokenizer_dir)
    digest = hashlib.sha256()
    for name in ("vocab.json", "merges.txt"):
        path = tokenizer_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing tokenizer file: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    tokenizer_sha256 = digest.hexdigest()

    errors = []
    if tokens != args.expected_tokens:
        errors.append(f"metadata tokens {tokens} != expected {args.expected_tokens}")
    if target != args.expected_tokens:
        errors.append(f"metadata target {target} != expected {args.expected_tokens}")
    if source_tokens != args.expected_tokens:
        errors.append(f"source token sum {source_tokens} != expected {args.expected_tokens}")
    if actual_bytes != expected_bytes:
        errors.append(f"binary bytes {actual_bytes} != expected {expected_bytes}")
    if int(meta.get("vocab_size", -1)) != args.expected_vocab_size:
        errors.append(f"metadata vocab_size {meta.get('vocab_size')} != expected {args.expected_vocab_size}")
    if meta.get("tokenizer_sha256") != tokenizer_sha256:
        errors.append("metadata tokenizer fingerprint does not match tokenizer/vocab.json and tokenizer/merges.txt")
    if errors:
        raise ValueError("; ".join(errors))

    print(
        json.dumps(
            {
                "status": "ok",
                "tokens": tokens,
                "bytes": actual_bytes,
                "sources": len(meta.get("sources", [])),
                "dtype": meta.get("dtype"),
                "vocab_size": meta.get("vocab_size"),
                "tokenizer_sha256": tokenizer_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
