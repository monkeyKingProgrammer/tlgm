import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import load_tokenizer  # noqa: E402


DEFAULT_SOURCES = [
    {
        "name": "smollm_fineweb_edu_dedup",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "fineweb-edu-dedup",
        "split": "train",
        "weight": 0.40,
        "field": "text",
    },
    {
        "name": "smollm_cosmopedia_v2",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "split": "train",
        "weight": 0.30,
        "field": "text",
    },
    {
        "name": "wikimedia_wikipedia_en",
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "weight": 0.30,
        "field": "text",
    },
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def source_quotas(target_tokens: int) -> list[dict]:
    quotas = []
    allocated = 0
    for source in DEFAULT_SOURCES[:-1]:
        quota = int(target_tokens * source["weight"])
        quotas.append({**source, "quota_tokens": quota})
        allocated += quota
    quotas.append({**DEFAULT_SOURCES[-1], "quota_tokens": target_tokens - allocated})
    return quotas


def text_from_row(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        return value.strip()
    for fallback in ("text", "content", "article", "body", "prompt"):
        value = row.get(fallback)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def write_ids(token_mem, offset: int, ids: list[int], target_tokens: int) -> int:
    if offset + len(ids) > target_tokens:
        ids = ids[: target_tokens - offset]
    if not ids:
        return offset
    token_mem[offset : offset + len(ids)] = np.asarray(ids, dtype=np.uint16)
    return offset + len(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=3_000_000_000)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_bin", default="data/processed/knowledge3b_tokens.bin")
    parser.add_argument("--meta", default="data/processed/knowledge3b_tokens_meta.json")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve(args.tokenizer_dir))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    output_bin = resolve(args.output_bin)
    meta_path = resolve(args.meta)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    quotas = source_quotas(args.target_tokens)
    token_mem = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
    offset = 0
    source_records = []

    try:
        for source in quotas:
            quota = int(source["quota_tokens"])
            source_start = offset
            source_target = min(args.target_tokens, source_start + quota)
            print("=" * 72)
            print(f"Source: {source['name']}")
            print(f"Dataset: {source['dataset']} / {source['config']}")
            print(f"Quota tokens: {quota:,}")
            print("=" * 72)

            ds = load_dataset(
                source["dataset"],
                source["config"],
                split=source["split"],
                streaming=True,
                cache_dir=args.hf_cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
            progress = tqdm(total=quota, unit="tok", desc=source["name"])
            rows = 0
            for row in ds:
                if offset >= source_target or offset >= args.target_tokens:
                    break
                text = text_from_row(row, source["field"])
                if len(text) < 80:
                    continue
                ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
                before = offset
                offset = write_ids(token_mem, offset, ids, min(source_target, args.target_tokens))
                progress.update(offset - before)
                rows += 1
            progress.close()
            source_records.append(
                {
                    "name": source["name"],
                    "dataset": source["dataset"],
                    "config": source["config"],
                    "weight": source["weight"],
                    "quota_tokens": quota,
                    "written_tokens": offset - source_start,
                    "rows_used": rows,
                }
            )
            token_mem.flush()
            if offset >= args.target_tokens:
                break
    finally:
        token_mem.flush()
        del token_mem

    final_size = offset * np.dtype(np.uint16).itemsize
    if output_bin.stat().st_size != final_size:
        with output_bin.open("r+b") as f:
            f.truncate(final_size)

    meta = {
        "tokens": offset,
        "target_tokens": args.target_tokens,
        "dtype": "uint16",
        "output_bin": str(output_bin),
        "sources": source_records,
        "note": "Knowledge-pretraining token stream generated by streaming Hugging Face datasets and tokenizing with the TLGM tokenizer.",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
