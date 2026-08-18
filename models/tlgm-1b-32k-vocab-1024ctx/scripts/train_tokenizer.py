import argparse
import hashlib
import json
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import load_tokenizer, train_byte_bpe  # noqa: E402
from tlgm.utils import atomic_json_save  # noqa: E402


SOURCES = [
    ("fineweb_edu_dedup", "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", 0.35),
    ("cosmopedia_v2", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 0.25),
    ("fineweb_general", "HuggingFaceFW/fineweb", "sample-10BT", 0.17),
    ("openwebmath", "open-web-math/open-web-math", None, 0.15),
    ("wikipedia_en", "wikimedia/wikipedia", "20231101.en", 0.08),
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


def dataset_revision(repository: str) -> str | None:
    try:
        return HfApi().dataset_info(repository).sha
    except Exception as error:
        print(f"Warning: could not resolve revision for {repository}: {error}", flush=True)
        return None


def stream_dataset(repository: str, config: str | None, revision: str | None, cache_dir: str | None):
    kwargs = {
        "path": repository,
        "split": "train",
        "streaming": True,
        "cache_dir": cache_dir,
    }
    if config is not None:
        kwargs["name"] = config
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(**kwargs)


def extract_text(row: dict) -> str:
    for field in ("text", "content", "article", "body"):
        value = row.get(field)
        if isinstance(value, str) and len(value.strip()) >= 80:
            return value.strip()
    return ""


def build_corpus(output_dir: Path, target_bytes: int, cache_dir: str | None, force: bool) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    allocated = 0
    for index, (name, repository, config, weight) in enumerate(SOURCES):
        quota = int(target_bytes * weight) if index < len(SOURCES) - 1 else target_bytes - allocated
        allocated += quota
        path = output_dir / f"{name}.txt"
        source_meta_path = output_dir / f"{name}.json"
        source_meta = json.loads(source_meta_path.read_text(encoding="utf-8")) if source_meta_path.exists() else {}
        revision = source_meta.get("revision") or dataset_revision(repository)
        if path.exists() and path.stat().st_size >= quota and not force:
            print(f"Reusing tokenizer source: {path}")
        else:
            temp = path.with_suffix(".txt.tmp")
            written = 0
            rows = 0
            dataset = stream_dataset(repository, config, revision, cache_dir)
            with temp.open("w", encoding="utf-8", newline="\n") as handle:
                progress = tqdm(total=quota, unit="B", unit_scale=True, desc=name)
                for row in dataset:
                    text = extract_text(row)
                    if not text:
                        continue
                    encoded = (text.replace("\x00", " ") + "\n").encode("utf-8")
                    remaining = quota - written
                    if len(encoded) > remaining:
                        encoded = encoded[:remaining]
                        text = encoded.decode("utf-8", errors="ignore")
                        encoded = text.encode("utf-8")
                    if not encoded:
                        break
                    handle.write(encoded.decode("utf-8"))
                    written += len(encoded)
                    rows += 1
                    progress.update(len(encoded))
                    if written >= quota:
                        break
                progress.close()
            if written < int(quota * 0.99):
                temp.unlink(missing_ok=True)
                raise RuntimeError(f"Tokenizer source {name} exhausted at {written:,} / {quota:,} bytes")
            temp.replace(path)
            atomic_json_save(
                {
                    "name": name,
                    "repository": repository,
                    "config": config,
                    "revision": revision,
                    "weight": weight,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                },
                source_meta_path,
            )
        records.append(
            {
                "name": name,
                "repository": repository,
                "config": config,
                "revision": source_meta.get("revision", revision),
                "weight": weight,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "path": str(path.relative_to(PROJECT_DIR)),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TLGM 32K byte-level BPE tokenizer from scratch.")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--corpus_bytes", type=int, default=4_000_000_000)
    parser.add_argument("--corpus_dir", default="data/processed/tokenizer_corpus_32k")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--manifest", default="tokenizer/manifest.json")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--force_corpus", action="store_true")
    args = parser.parse_args()

    if not 256 < args.vocab_size <= 65_536:
        parser.error("vocab_size must be between 257 and 65,536 for uint16 token storage")
    records = build_corpus(resolve(args.corpus_dir), args.corpus_bytes, args.hf_cache_dir, args.force_corpus)
    tokenizer_dir = resolve(args.tokenizer_dir)
    train_byte_bpe(
        [str(resolve(record["path"])) for record in records],
        tokenizer_dir,
        vocab_size=args.vocab_size,
        min_frequency=2,
    )
    tokenizer = load_tokenizer(tokenizer_dir)
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f"Tokenizer has {tokenizer.get_vocab_size()} entries, expected {args.vocab_size}")
    special_tokens = {token: tokenizer.token_to_id(token) for token in ("<pad>", "<bos>", "<eos>", "<unk>")}
    expected_special = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    if special_tokens != expected_special:
        raise RuntimeError(f"Unexpected special-token IDs: {special_tokens}")
    manifest = {
        "schema_version": 1,
        "algorithm": "byte-level BPE",
        "trained_from_scratch": True,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_tokens,
        "corpus_target_bytes": args.corpus_bytes,
        "sources": records,
        "files": {
            name: {"bytes": (tokenizer_dir / name).stat().st_size, "sha256": sha256_file(tokenizer_dir / name)}
            for name in ("vocab.json", "merges.txt")
        },
    }
    atomic_json_save(manifest, resolve(args.manifest))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
