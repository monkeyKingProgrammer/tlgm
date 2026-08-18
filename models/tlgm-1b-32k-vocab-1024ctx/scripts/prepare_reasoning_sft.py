import argparse
import hashlib
import heapq
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from prepare_logic_sft import build_rows


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
DECONTAM_NGRAM_SIZE = 13


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def chat(user: str, assistant: str) -> dict:
    return {"conversations": [
        {"role": "user", "content": user.strip()},
        {"role": "assistant", "content": assistant.strip()},
    ]}


def valid(row: dict) -> bool:
    messages = row.get("conversations", [])
    return any(m.get("role") == "user" and str(m.get("content", "")).strip() for m in messages) and any(
        m.get("role") == "assistant" and str(m.get("content", "")).strip() for m in messages
    )


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, examples: int) -> dict:
    return {
        "path": str(path),
        "examples": examples,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_atomic(path: Path, rows, name: str, force: bool = False) -> int:
    if path.exists() and not force:
        count = count_lines(path)
        print(f"{name}: reusing {count:,} prepared records", flush=True)
        return count
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for item in rows:
            if valid(item):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count % 10_000 == 0:
                    print(f"{name}: prepared {count:,} records", flush=True)
    temp.replace(path)
    return count


def limited(dataset, limit: int):
    for index, item in enumerate(dataset):
        if index >= limit:
            break
        yield item


def openmath_rows(limit: int, seed: int, revision: str | None):
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True, revision=revision)
    for item in limited(dataset.shuffle(seed=seed, buffer_size=100_000), limit):
        yield chat(str(item["problem"]), str(item["generated_solution"]))


def openr1_rows(limit: int, seed: int, revision: str | None):
    dataset = load_dataset(
        "open-r1/OpenR1-Math-220k", "default", split="train", streaming=True, revision=revision
    )
    for item in limited(dataset.shuffle(seed=seed, buffer_size=5_000), limit):
        yield chat(str(item["problem"]), str(item["solution"]))


def ultrachat_rows(limit: int, seed: int, revision: str | None):
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True, revision=revision
    )
    for item in limited(dataset.shuffle(seed=seed, buffer_size=10_000), limit):
        messages = []
        for message in item.get("messages", []):
            role = str(message.get("role", "user"))
            if role in {"human", "prompter"}:
                role = "user"
            elif role in {"gpt", "bot"}:
                role = "assistant"
            messages.append({"role": role, "content": str(message.get("content", "")).strip()})
        yield {"conversations": messages}


def existing_chat_rows(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def answer_index(answer, labels: list[str]) -> int:
    answer = str(answer).strip().upper()
    return [str(label).strip().upper() for label in labels].index(answer)


def science_rows(revisions: dict[str, str | None]):
    for config in ("ARC-Easy", "ARC-Challenge"):
        for item in load_dataset("allenai/ai2_arc", config, split="train", revision=revisions["allenai/ai2_arc"]):
            choices = item["choices"]
            index = answer_index(item["answerKey"], choices["label"])
            options = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
            yield chat(
                f"Choose the correct answer and explain briefly.\nQuestion: {item['question']}\n{options}",
                f"The correct answer is {choices['label'][index]}: {choices['text'][index]}.",
            )
    for item in load_dataset(
        "allenai/openbookqa", "main", split="train", revision=revisions["allenai/openbookqa"]
    ):
        choices = item["choices"]
        index = answer_index(item["answerKey"], choices["label"])
        options = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
        yield chat(
            f"Choose the correct answer and explain briefly.\nQuestion: {item['question_stem']}\n{options}",
            f"The correct answer is {choices['label'][index]}: {choices['text'][index]}.",
        )
    for item in load_dataset("google/boolq", split="train", revision=revisions["google/boolq"]):
        answer = "Yes" if item["answer"] else "No"
        yield chat(
            f"Read the passage and answer Yes or No.\nPassage: {item['passage']}\nQuestion: {item['question']}?",
            f"{answer}.",
        )


def normalized_messages(row: dict) -> list[dict]:
    messages = []
    for message in row.get("conversations", []):
        role = str(message.get("role", "user")).strip().lower()
        if role in {"human", "prompter"}:
            role = "user"
        elif role in {"gpt", "bot"}:
            role = "assistant"
        content = str(message.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


def target_records(row: dict, source: str):
    messages = normalized_messages(row)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = messages[: index + 1]
        canonical_messages = [
            {"role": item["role"], "content": " ".join(item["content"].split())}
            for item in prefix
        ]
        canonical = json.dumps(canonical_messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        record_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        yield {
            "id": record_id,
            "source": source,
            "target_index": index,
            "conversations": prefix,
        }


def word_ngrams(text: str, size: int = DECONTAM_NGRAM_SIZE):
    words = re.findall(r"[a-z0-9]+", text.lower())
    for index in range(len(words) - size + 1):
        joined = " ".join(words[index : index + size]).encode("utf-8")
        yield int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


def benchmark_decontamination_ngrams() -> set[int]:
    specs = (
        ("allenai/ai2_arc", "ARC-Easy", "test", "question"),
        ("allenai/ai2_arc", "ARC-Challenge", "test", "question"),
        ("Rowan/hellaswag", None, "validation", "ctx"),
        ("lighteval/piqa", "plain_text", "validation", "goal"),
        ("google/boolq", None, "validation", "question"),
        ("allenai/winogrande", "winogrande_xl", "validation", "sentence"),
        ("allenai/openbookqa", "main", "test", "question_stem"),
        ("truthfulqa/truthful_qa", "multiple_choice", "validation", "question"),
    )
    result = set()
    for dataset_name, config, split, field in specs:
        dataset = load_dataset(dataset_name, config, split=split)
        for row in dataset:
            result.update(word_ngrams(str(row.get(field, ""))))
    print(f"Benchmark decontamination index: {len(result):,} unique {DECONTAM_NGRAM_SIZE}-grams", flush=True)
    return result


def contaminated(record: dict, blocked_ngrams: set[int]) -> bool:
    if not blocked_ngrams:
        return False
    user_text = "\n".join(
        message["content"] for message in record["conversations"] if message["role"] == "user"
    )
    return any(value in blocked_ngrams for value in word_ngrams(user_text))


def expand_and_deduplicate(
    raw_specs: list[tuple[str, Path]], target_dir: Path, blocked_ngrams: set[int]
) -> tuple[list[Path], list[int], int, int]:
    seen = set()
    target_paths = []
    counts = []
    duplicates = 0
    contamination_removed = 0
    for source, raw_path in raw_specs:
        output = target_dir / f"{source}.jsonl"
        temp = output.with_suffix(".jsonl.tmp")
        count = 0
        with raw_path.open("r", encoding="utf-8") as source_handle, temp.open("w", encoding="utf-8") as target:
            for line in source_handle:
                if not line.strip():
                    continue
                for record in target_records(json.loads(line), source):
                    if contaminated(record, blocked_ngrams):
                        contamination_removed += 1
                        continue
                    if record["id"] in seen:
                        duplicates += 1
                        continue
                    seen.add(record["id"])
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        temp.replace(output)
        target_paths.append(output)
        counts.append(count)
        print(f"{source}: {count:,} unique assistant targets", flush=True)
    return target_paths, counts, duplicates, contamination_removed


def allocate_validation(counts: list[int], total: int) -> list[int]:
    total = min(total, max(0, sum(counts) - len(counts)))
    if total == 0:
        return [0] * len(counts)
    exact = [total * count / sum(counts) for count in counts]
    result = [min(count, int(value)) for count, value in zip(counts, exact)]
    remainder_order = sorted(range(len(counts)), key=lambda i: exact[i] - result[i], reverse=True)
    for index in remainder_order:
        if sum(result) >= total:
            break
        if result[index] < counts[index]:
            result[index] += 1
    return result


def validation_ids(path: Path, quota: int) -> set[str]:
    if quota <= 0:
        return set()
    heap = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_id = json.loads(line)["id"]
            score = int(hashlib.sha256(("validation:" + record_id).encode()).hexdigest(), 16)
            item = (-score, record_id)
            if len(heap) < quota:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
    return {record_id for _, record_id in heap}


def split_targets(paths: list[Path], counts: list[int], work_dir: Path, validation_output: Path, validation_size: int):
    quotas = allocate_validation(counts, validation_size)
    train_paths, train_counts = [], []
    validation_temp = validation_output.with_suffix(validation_output.suffix + ".tmp")
    with validation_temp.open("w", encoding="utf-8") as validation_handle:
        for path, count, quota in zip(paths, counts, quotas):
            selected = validation_ids(path, quota)
            train_path = work_dir / f"{path.stem}.train.jsonl"
            train_temp = train_path.with_suffix(".jsonl.tmp")
            train_count = 0
            validation_count = 0
            with path.open("r", encoding="utf-8") as source, train_temp.open("w", encoding="utf-8") as train:
                for line in source:
                    record = json.loads(line)
                    if record["id"] in selected:
                        validation_handle.write(line)
                        validation_count += 1
                    else:
                        train.write(line)
                        train_count += 1
            if validation_count != quota or train_count + validation_count != count:
                raise RuntimeError(f"Split count mismatch for {path}")
            train_temp.replace(train_path)
            train_paths.append(train_path)
            train_counts.append(train_count)
    validation_temp.replace(validation_output)
    return train_paths, train_counts, quotas


def merge_weighted(paths: list[Path], counts: list[int], output: Path, seed: int) -> None:
    rng = random.Random(seed)
    handles = [path.open("r", encoding="utf-8") for path in paths]
    remaining = counts[:]
    temp = output.with_suffix(output.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as target:
            total_remaining = sum(remaining)
            while total_remaining:
                pick = rng.randrange(total_remaining)
                running = 0
                source_index = 0
                for source_index, count in enumerate(remaining):
                    running += count
                    if pick < running:
                        break
                line = handles[source_index].readline()
                if not line:
                    raise RuntimeError(f"Unexpected EOF while merging {paths[source_index]}")
                target.write(line)
                remaining[source_index] -= 1
                total_remaining -= 1
    finally:
        for handle in handles:
            handle.close()
    temp.replace(output)


def dataset_revision(dataset: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(dataset).sha
    except Exception as error:
        print(f"Warning: could not resolve revision for {dataset}: {error}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/sft_reasoning_32k.jsonl")
    parser.add_argument("--validation_output", default="data/processed/sft_reasoning_validation_32k.jsonl")
    parser.add_argument("--work_dir", default="data/processed/reasoning_sources_32k")
    parser.add_argument("--validation_size", type=int, default=5_000)
    parser.add_argument("--openmath", type=int, default=1_000_000)
    parser.add_argument("--openr1", type=int, default=200_000)
    parser.add_argument("--ultrachat", type=int, default=200_000)
    parser.add_argument("--existing_chat", type=int, default=200_000)
    parser.add_argument("--logic_repeats", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--force_sources", action="store_true")
    parser.add_argument("--skip_decontamination", action="store_true")
    args = parser.parse_args()

    output = resolve(args.output)
    validation_output = resolve(args.validation_output)
    work_dir = resolve(args.work_dir)
    target_dir = work_dir / "targets"
    work_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    repositories = (
        "nvidia/OpenMathInstruct-2",
        "open-r1/OpenR1-Math-220k",
        "HuggingFaceH4/ultrachat_200k",
        "allenai/ai2_arc",
        "allenai/openbookqa",
        "google/boolq",
    )
    revisions = {name: dataset_revision(name) for name in repositories}
    specs = [
        ("openmath", lambda: openmath_rows(args.openmath, args.seed, revisions["nvidia/OpenMathInstruct-2"])),
        ("openr1", lambda: openr1_rows(args.openr1, args.seed + 1, revisions["open-r1/OpenR1-Math-220k"])),
        ("ultrachat", lambda: ultrachat_rows(args.ultrachat, args.seed + 2, revisions["HuggingFaceH4/ultrachat_200k"])),
        ("existing_chat", lambda: existing_chat_rows(resolve("data/processed/sft_chat_32k.jsonl"), args.existing_chat)),
        ("verified_logic", lambda: iter(build_rows(args.seed + 3, args.logic_repeats))),
        ("science_reading", lambda: science_rows(revisions)),
    ]
    raw_specs = []
    for name, factory in specs:
        path = work_dir / f"{name}.raw.jsonl"
        legacy_path = resolve("data/processed/reasoning_sources") / f"{name}.jsonl"
        if not path.exists() and legacy_path.exists() and not args.force_sources:
            path = legacy_path
            count = count_lines(path)
            print(f"{name}: reusing legacy source artifact {path}", flush=True)
        else:
            count = write_atomic(path, factory(), name, force=args.force_sources)
        raw_specs.append((name, path))
        print(f"{name}: {count:,} source conversations", flush=True)

    blocked_ngrams = set() if args.skip_decontamination else benchmark_decontamination_ngrams()
    target_paths, target_counts, duplicate_count, contamination_removed = expand_and_deduplicate(
        raw_specs, target_dir, blocked_ngrams
    )
    train_paths, train_counts, validation_counts = split_targets(
        target_paths, target_counts, work_dir, validation_output, args.validation_size
    )
    merge_weighted(train_paths, train_counts, output, args.seed + 4)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "deduplication": "SHA-256 of normalized conversation prefix through each target assistant turn",
        "duplicates_removed": duplicate_count,
        "benchmark_decontamination": {
            "enabled": not args.skip_decontamination,
            "ngram_size": DECONTAM_NGRAM_SIZE,
            "indexed_ngrams": len(blocked_ngrams),
            "targets_removed": contamination_removed,
        },
        "sources": [
            {
                "name": name,
                "unique_targets": total,
                "train_targets": train,
                "validation_targets": validation,
            }
            for (name, _), total, train, validation in zip(raw_specs, target_counts, train_counts, validation_counts)
        ],
        "upstream_revisions": revisions,
        "train": artifact(output, sum(train_counts)),
        "validation": artifact(validation_output, sum(validation_counts)),
    }
    manifest_path = output.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
