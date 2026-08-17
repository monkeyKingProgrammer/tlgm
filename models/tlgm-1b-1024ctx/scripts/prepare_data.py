import argparse
import gzip
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


def iter_jsonl_files(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
        yield from sorted(path.glob("*.jsonl.gz"))
    else:
        yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", default="../minimind_120m_tiny_chatgpt/data/processed/pretrain_2b")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_bin", default="data/processed/pretrain_tokens.bin")
    parser.add_argument("--meta", default="data/processed/pretrain_tokens_meta.json")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve(args.tokenizer_dir))
    output_bin = resolve(args.output_bin)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    token_mem = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
    offset = 0
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    for file in iter_jsonl_files(resolve(args.pretrain_dir)):
        opener = gzip.open if file.suffix == ".gz" else open
        with opener(file, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc=file.name):
                if not line.strip():
                    continue
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
                if offset + len(ids) > args.target_tokens:
                    ids = ids[: args.target_tokens - offset]
                if not ids:
                    break
                token_mem[offset : offset + len(ids)] = np.asarray(ids, dtype=np.uint16)
                offset += len(ids)
                if offset >= args.target_tokens:
                    break
        if offset >= args.target_tokens:
            break
    token_mem.flush()
    del token_mem
    final_size = offset * np.dtype(np.uint16).itemsize
    if output_bin.stat().st_size != final_size:
        with output_bin.open("r+b") as f:
            f.truncate(final_size)
    meta = {"tokens": offset, "target_tokens": args.target_tokens, "dtype": "uint16", "output_bin": str(output_bin)}
    with resolve(args.meta).open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
