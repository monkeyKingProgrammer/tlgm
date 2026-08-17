import argparse
import gzip
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class SourceSpec:
    name: str
    weight: float
    dataset: str
    config: str | None
    split: str
    text_keys: tuple[str, ...]
    trust_remote_code: bool = False


SOURCES = [
    SourceSpec("fineweb_edu", 0.40, "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", "train", ("text", "content")),
    SourceSpec("cosmopedia", 0.20, "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", ("text", "prompt", "completion")),
    SourceSpec("wikipedia", 0.15, "wikimedia/wikipedia", "20231101.en", "train", ("text",)),
    SourceSpec("tinystories", 0.10, "roneneldan/TinyStories", None, "train", ("text", "story")),
    SourceSpec("fineweb_broad", 0.10, "HuggingFaceFW/fineweb", "sample-10BT", "train", ("text", "content")),
    SourceSpec("code", 0.05, "codeparrot/github-code-clean", None, "train", ("code", "text", "content"), True),
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text(row: dict, keys: tuple[str, ...]) -> str:
    parts = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:
        for value in row.values():
            if isinstance(value, str) and len(value) > 200:
                parts.append(value.strip())
                break
    return normalize("\n\n".join(parts))


def load_stream(spec: SourceSpec):
    kwargs = {"split": spec.split, "streaming": True}
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if spec.config:
        return load_dataset(spec.dataset, spec.config, **kwargs)
    return load_dataset(spec.dataset, **kwargs)


class ShardWriter:
    def __init__(self, output_dir: Path, max_rows: int, compress: bool):
        self.output_dir = output_dir
        self.max_rows = max_rows
        self.compress = compress
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.rows_in_shard = 0
        self.handle = None
        self.open_next()

    def open_next(self) -> None:
        if self.handle:
            self.handle.close()
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        path = self.output_dir / f"part-{self.index:05d}{suffix}"
        self.handle = gzip.open(path, "wt", encoding="utf-8") if self.compress else path.open("w", encoding="utf-8")
        self.index += 1
        self.rows_in_shard = 0

    def write(self, row: dict) -> None:
        if self.rows_in_shard >= self.max_rows:
            self.open_next()
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows_in_shard += 1

    def close(self) -> None:
        if self.handle:
            self.handle.close()
            self.handle = None


def source_quota_tokens(total_tokens: int) -> dict[str, int]:
    quotas = {spec.name: int(total_tokens * spec.weight) for spec in SOURCES}
    diff = total_tokens - sum(quotas.values())
    quotas[SOURCES[0].name] += diff
    return quotas


def prepare_source(spec: SourceSpec, quota: int, tokenizer, writer: ShardWriter, min_chars: int, max_chars: int, seed: int) -> tuple[int, int]:
    token_count = 0
    rows = 0
    try:
        stream = load_stream(spec)
    except Exception as exc:
        print(f"FAILED to open {spec.name}: {exc}", file=sys.stderr)
        return rows, token_count

    stream = stream.shuffle(seed=seed, buffer_size=10_000)
    progress = tqdm(total=quota, desc=spec.name, unit="tok")
    for row in stream:
        text = extract_text(dict(row), spec.text_keys)
        if len(text) < min_chars:
            continue
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        n_tokens = len(token_ids)
        if n_tokens < 32:
            continue
        writer.write({"text": text, "source": spec.name})
        rows += 1
        token_count += n_tokens
        progress.update(n_tokens)
        if token_count >= quota:
            break
    progress.close()
    return rows, token_count


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/processed/pretrain_2b")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--tokenizer_path", default="../MiniMind/model")
    parser.add_argument("--max_rows_per_shard", type=int, default=100_000)
    parser.add_argument("--min_chars", type=int, default=120)
    parser.add_argument("--max_chars", type=int, default=12000)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--seed", type=int, default=120)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = resolve(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(resolve(args.tokenizer_path))
    writer = ShardWriter(output_dir, args.max_rows_per_shard, args.compress)
    quotas = source_quota_tokens(args.target_tokens)
    manifest = {
        "target_tokens": args.target_tokens,
        "mix": [{"name": spec.name, "weight": spec.weight, "quota_tokens": quotas[spec.name], "dataset": spec.dataset, "config": spec.config} for spec in SOURCES],
        "sources": {},
    }

    try:
        for spec in SOURCES:
            rows, tokens = prepare_source(spec, quotas[spec.name], tokenizer, writer, args.min_chars, args.max_chars, args.seed)
            manifest["sources"][spec.name] = {"rows": rows, "tokens": tokens}
            write_manifest(output_dir / "manifest.json", manifest)
    finally:
        writer.close()

    manifest["actual_tokens"] = sum(v["tokens"] for v in manifest["sources"].values())
    manifest["actual_rows"] = sum(v["rows"] for v in manifest["sources"].values())
    write_manifest(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
