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
from tlgm.utils import atomic_json_save  # noqa: E402


SOURCES = [
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
        "weight": 0.25,
        "field": "text",
    },
    {
        "name": "wikimedia_wikipedia_en",
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "split": "train",
        "weight": 0.20,
        "field": "text",
    },
    {
        "name": "fineweb_edu_sample10bt",
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "split": "train",
        "weight": 0.15,
        "field": "text",
    },
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def quotas(target_tokens: int) -> list[dict]:
    result = []
    allocated = 0
    for source in SOURCES[:-1]:
        quota = int(target_tokens * source["weight"])
        result.append({**source, "quota_tokens": quota})
        allocated += quota
    result.append({**SOURCES[-1], "quota_tokens": target_tokens - allocated})
    return result


def get_text(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for fallback in ("text", "content", "article", "body"):
        value = row.get(fallback)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=20_000_000_000)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_bin", default="data/processed/pretrain20b_tokens.bin")
    parser.add_argument("--meta", default="data/processed/pretrain20b_tokens_meta.json")
    parser.add_argument(
        "--resume_tokens",
        type=int,
        default=None,
        help="Resume writing at this already-complete token offset. Requires an existing full-size bin.",
    )
    parser.add_argument(
        "--state",
        default="data/processed/pretrain20b_tokens_progress.json",
        help="Progress file updated during tokenization so a later restart can resume safely.",
    )
    parser.add_argument("--hf_cache_dir", default=None)
    args = parser.parse_args()

    tokenizer = load_tokenizer(resolve(args.tokenizer_dir))
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    output_bin = resolve(args.output_bin)
    meta_path = resolve(args.meta)
    state_path = resolve(args.state)
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    expected_size = args.target_tokens * np.dtype(np.uint16).itemsize
    resume_tokens = args.resume_tokens
    if resume_tokens is None and state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            resume_tokens = int(json.load(f).get("tokens", 0))
    resume_tokens = int(resume_tokens or 0)
    if not 0 <= resume_tokens <= args.target_tokens:
        raise ValueError("resume_tokens must be between zero and target_tokens")

    if resume_tokens:
        if not output_bin.exists() or output_bin.stat().st_size < expected_size:
            raise RuntimeError("Cannot resume: the preallocated token bin is missing or too small.")
        token_mem = np.memmap(output_bin, dtype=np.uint16, mode="r+", shape=(args.target_tokens,))
        print(f"Resuming tokenization at {resume_tokens:,} / {args.target_tokens:,} tokens.")
    else:
        token_mem = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
    offset = resume_tokens
    last_state_offset = offset
    records = []
    try:
        for source in quotas(args.target_tokens):
            quota = int(source["quota_tokens"])
            source_start = sum(item["quota_tokens"] for item in quotas(args.target_tokens)[: len(records)])
            target = min(args.target_tokens, source_start + quota)
            if offset >= target:
                records.append(
                    {
                        "name": source["name"],
                        "dataset": source["dataset"],
                        "config": source["config"],
                        "weight": source["weight"],
                        "quota_tokens": quota,
                        "written_tokens": quota,
                        "rows_used": None,
                    }
                )
                continue
            already_in_source = max(0, offset - source_start)
            print("=" * 72)
            print(f"Source: {source['name']}")
            print(f"Dataset: {source['dataset']} / {source['config']}")
            print(f"Quota: {quota:,} tokens")
            print("=" * 72)
            ds = load_dataset(
                source["dataset"],
                source["config"],
                split=source["split"],
                streaming=True,
                cache_dir=args.hf_cache_dir,
            )
            rows = 0
            progress = tqdm(total=quota, initial=already_in_source, unit="tok", desc=source["name"])
            skip_tokens = already_in_source
            for row in ds:
                if offset >= target or offset >= args.target_tokens:
                    break
                text = get_text(row, source["field"])
                if len(text) < 80:
                    continue
                ids = [bos_id] + tokenizer.encode(text).ids + [eos_id]
                if skip_tokens:
                    if len(ids) <= skip_tokens:
                        skip_tokens -= len(ids)
                        continue
                    # A previous run was interrupted inside this example.
                    ids = ids[skip_tokens:]
                    skip_tokens = 0
                if offset + len(ids) > target:
                    ids = ids[: target - offset]
                if not ids:
                    break
                token_mem[offset : offset + len(ids)] = np.asarray(ids, dtype=np.uint16)
                offset += len(ids)
                rows += 1
                progress.update(len(ids))
                if offset - last_state_offset >= 10_000_000:
                    token_mem.flush()
                    atomic_json_save({"tokens": offset, "target_tokens": args.target_tokens}, state_path)
                    last_state_offset = offset
            progress.close()
            token_mem.flush()
            records.append(
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
        "sources": records,
    }
    atomic_json_save(meta, meta_path)
    state_path.unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
