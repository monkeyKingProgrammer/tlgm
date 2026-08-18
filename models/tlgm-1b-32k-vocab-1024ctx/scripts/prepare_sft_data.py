import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(messages: list[dict]) -> dict | None:
    result = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        if role in {"human", "prompter"}:
            role = "user"
        elif role in {"gpt", "bot"}:
            role = "assistant"
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and content:
            result.append({"role": role, "content": content})
    if not any(item["role"] == "user" for item in result):
        return None
    if not any(item["role"] == "assistant" for item in result):
        return None
    return {"conversations": result}


def ultrachat_rows(limit: int, dataset_revision: str | None):
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, revision=dataset_revision
    )
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        converted = normalized(row.get("messages", []))
        if converted:
            yield converted


def dolly_rows(limit: int, dataset_revision: str | None):
    dataset = load_dataset(
        "databricks/databricks-dolly-15k", split="train", streaming=True, revision=dataset_revision
    )
    for index, row in enumerate(dataset):
        if index >= limit:
            break
        instruction = str(row.get("instruction", "")).strip()
        context = str(row.get("context", "")).strip()
        response = str(row.get("response", "")).strip()
        if context:
            instruction = f"{instruction}\n\nContext:\n{context}"
        converted = normalized(
            [{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]
        )
        if converted:
            yield converted


def canonical_hash(row: dict) -> str:
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def revision(repository: str) -> str | None:
    try:
        return HfApi().dataset_info(repository).sha
    except Exception as error:
        print(f"Warning: could not resolve revision for {repository}: {error}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare broad, deduplicated chat SFT data.")
    parser.add_argument("--output", default="data/processed/sft_chat_32k.jsonl")
    parser.add_argument("--manifest", default="data/processed/sft_chat_32k.manifest.json")
    parser.add_argument("--ultrachat", type=int, default=200_000)
    parser.add_argument("--dolly", type=int, default=15_000)
    args = parser.parse_args()

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    seen = set()
    source_counts = {}
    duplicate_count = 0
    revisions = {
        "ultrachat": revision("HuggingFaceH4/ultrachat_200k"),
        "dolly": revision("databricks/databricks-dolly-15k"),
    }
    sources = [
        ("ultrachat", ultrachat_rows(args.ultrachat, revisions["ultrachat"])),
        ("dolly", dolly_rows(args.dolly, revisions["dolly"])),
    ]
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for source, rows in sources:
            count = 0
            for row in tqdm(rows, desc=source):
                row_hash = canonical_hash(row)
                if row_hash in seen:
                    duplicate_count += 1
                    continue
                seen.add(row_hash)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            source_counts[source] = count
    temp.replace(output)
    manifest = {
        "schema_version": 1,
        "deduplication": "SHA-256 of normalized conversation JSON",
        "duplicates_removed": duplicate_count,
        "sources": [
            {
                "name": "HuggingFaceH4/ultrachat_200k",
                "split": "train_sft",
                "revision": revisions["ultrachat"],
                "examples": source_counts["ultrachat"],
            },
            {
                "name": "databricks/databricks-dolly-15k",
                "split": "train",
                "revision": revisions["dolly"],
                "examples": source_counts["dolly"],
            },
        ],
        "train": {
            "path": str(output.relative_to(PROJECT_DIR)),
            "examples": sum(source_counts.values()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }
    manifest_path = resolve(args.manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
