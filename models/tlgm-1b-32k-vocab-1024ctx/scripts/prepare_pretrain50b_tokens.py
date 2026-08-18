import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import load_tokenizer  # noqa: E402
from tlgm.utils import atomic_json_save  # noqa: E402


SOURCES = [
    {"name": "fineweb_edu_dedup", "dataset": "HuggingFaceTB/smollm-corpus", "config": "fineweb-edu-dedup", "weight": 0.35},
    {"name": "cosmopedia_v2", "dataset": "HuggingFaceTB/smollm-corpus", "config": "cosmopedia-v2", "weight": 0.25},
    {"name": "fineweb_general", "dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "weight": 0.17},
    {"name": "openwebmath", "dataset": "open-web-math/open-web-math", "config": None, "weight": 0.15},
    {"name": "wikipedia_en", "dataset": "wikimedia/wikipedia", "config": "20231101.en", "weight": 0.08},
]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(tokenizer_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("vocab.json", "merges.txt"):
        path = tokenizer_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing tokenizer file: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_plan(target_tokens: int) -> list[dict]:
    revisions = {}
    api = HfApi()
    for source in SOURCES:
        repository = source["dataset"]
        if repository not in revisions:
            revisions[repository] = api.dataset_info(repository).sha
    plan = []
    allocated = 0
    for index, source in enumerate(SOURCES):
        quota = int(target_tokens * source["weight"]) if index < len(SOURCES) - 1 else target_tokens - allocated
        allocated += quota
        plan.append({**source, "revision": revisions[source["dataset"]], "quota_tokens": quota})
    return plan


def plan_fingerprint(plan: list[dict], target_tokens: int, tokenizer_sha256: str) -> str:
    payload = {"target_tokens": target_tokens, "tokenizer_sha256": tokenizer_sha256, "sources": plan}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def extract_text(row: dict) -> str:
    for field in ("text", "content", "article", "body"):
        value = row.get(field)
        if isinstance(value, str) and len(value.strip()) >= 80:
            return value.strip()
    return ""


def open_stream(source: dict, cache_dir: str | None):
    kwargs = {
        "path": source["dataset"],
        "split": "train",
        "streaming": True,
        "revision": source["revision"],
        "cache_dir": cache_dir,
    }
    if source["config"] is not None:
        kwargs["name"] = source["config"]
    return load_dataset(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the reproducible TLGM 32K 50B-token stream.")
    parser.add_argument("--target_tokens", type=int, default=50_000_000_000)
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--output_bin", default="data/processed/pretrain50b_32k_tokens.bin")
    parser.add_argument("--meta", default="data/processed/pretrain50b_32k_tokens_meta.json")
    parser.add_argument("--state", default="data/processed/pretrain50b_32k_tokens_progress.json")
    parser.add_argument("--hf_cache_dir", default=None)
    args = parser.parse_args()

    if not 0 < args.target_tokens:
        parser.error("target_tokens must be positive")
    tokenizer_dir = resolve(args.tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    if tokenizer.get_vocab_size() != 32_000:
        raise ValueError(f"Expected a 32,000-entry tokenizer, found {tokenizer.get_vocab_size()}")
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer_dir)
    output_bin = resolve(args.output_bin)
    meta_path = resolve(args.meta)
    state_path = resolve(args.state)
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = state.get("sources") or source_plan(args.target_tokens)
    fingerprint = plan_fingerprint(plan, args.target_tokens, tokenizer_sha256)
    if state:
        if state.get("plan_sha256") != fingerprint:
            raise RuntimeError("Existing progress belongs to a different tokenizer, source revision, or token target")
    offset = int(state.get("tokens", 0))
    if not 0 <= offset <= args.target_tokens:
        raise ValueError("Progress token offset is invalid")
    expected_size = args.target_tokens * np.dtype(np.uint16).itemsize
    if offset:
        if not output_bin.is_file() or output_bin.stat().st_size != expected_size:
            raise RuntimeError("Cannot resume: the preallocated 100GB token binary is missing or has the wrong size")
        token_mem = np.memmap(output_bin, dtype=np.uint16, mode="r+", shape=(args.target_tokens,))
        print(f"Resuming at {offset:,} / {args.target_tokens:,} tokens")
    else:
        token_mem = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=(args.target_tokens,))
        atomic_json_save(
            {
                "schema_version": 1,
                "tokens": 0,
                "target_tokens": args.target_tokens,
                "plan_sha256": fingerprint,
                "sources": plan,
            },
            state_path,
        )

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    records = []
    source_start = 0
    last_saved = offset
    try:
        for source in plan:
            quota = int(source["quota_tokens"])
            source_end = source_start + quota
            if offset >= source_end:
                records.append({**source, "written_tokens": quota, "rows_used": None})
                source_start = source_end
                continue
            if offset < source_start:
                raise RuntimeError("Progress offset falls before the current source boundary")
            skip_tokens = offset - source_start
            rows = 0
            print(f"Source {source['name']}: {quota:,} tokens at revision {source['revision']}")
            progress = tqdm(total=quota, initial=skip_tokens, unit="tok", desc=source["name"])
            for row in open_stream(source, args.hf_cache_dir):
                if offset >= source_end:
                    break
                text = extract_text(row)
                if not text:
                    continue
                ids = [bos_id, *tokenizer.encode(text).ids, eos_id]
                if skip_tokens:
                    if len(ids) <= skip_tokens:
                        skip_tokens -= len(ids)
                        continue
                    ids = ids[skip_tokens:]
                    skip_tokens = 0
                ids = ids[: source_end - offset]
                if not ids:
                    break
                if max(ids) >= 65_536:
                    raise ValueError("Tokenizer emitted an ID that cannot be represented as uint16")
                token_mem[offset : offset + len(ids)] = np.asarray(ids, dtype=np.uint16)
                offset += len(ids)
                rows += 1
                progress.update(len(ids))
                if offset - last_saved >= 10_000_000:
                    token_mem.flush()
                    atomic_json_save(
                        {
                            "schema_version": 1,
                            "tokens": offset,
                            "target_tokens": args.target_tokens,
                            "plan_sha256": fingerprint,
                            "sources": plan,
                        },
                        state_path,
                    )
                    last_saved = offset
            progress.close()
            if offset != source_end:
                raise RuntimeError(f"Source {source['name']} exhausted at {offset - source_start:,} / {quota:,} tokens")
            records.append({**source, "written_tokens": quota, "rows_used": rows})
            source_start = source_end
    finally:
        token_mem.flush()
        del token_mem

    if offset != args.target_tokens or output_bin.stat().st_size != expected_size:
        raise RuntimeError("Token stream did not complete at the exact requested size")
    meta = {
        "schema_version": 1,
        "tokens": offset,
        "target_tokens": args.target_tokens,
        "dtype": "uint16",
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": tokenizer_sha256,
        "plan_sha256": fingerprint,
        "output_bin": str(output_bin.relative_to(PROJECT_DIR)),
        "output_bytes": output_bin.stat().st_size,
        "sources": records,
    }
    atomic_json_save(meta, meta_path)
    state_path.unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
