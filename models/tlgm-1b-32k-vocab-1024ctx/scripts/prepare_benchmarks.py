import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]

TASKS = {
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "config": None,
        "split": "validation",
    },
    "piqa": {
        "dataset": "lighteval/piqa",
        "config": "plain_text",
        "split": "validation",
    },
    "boolq": {
        "dataset": "google/boolq",
        "config": None,
        "split": "validation",
    },
    "winogrande": {
        "dataset": "allenai/winogrande",
        "config": "winogrande_xl",
        "split": "validation",
    },
    "openbookqa": {
        "dataset": "allenai/openbookqa",
        "config": "main",
        "split": "test",
    },
    "truthfulqa_mc1": {
        "dataset": "truthfulqa/truthful_qa",
        "config": "multiple_choice",
        "split": "validation",
    },
}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def answer_index(answer, labels: list[str]) -> int:
    value = str(answer).strip()
    if value in labels:
        return labels.index(value)
    upper = value.upper()
    upper_labels = [str(label).upper() for label in labels]
    if upper in upper_labels:
        return upper_labels.index(upper)
    if value.isdigit():
        numeric = int(value)
        if 0 <= numeric < len(labels):
            return numeric
        if 1 <= numeric <= len(labels):
            return numeric - 1
    raise ValueError(f"Cannot map answer {answer!r} to labels {labels!r}")


def normalize_arc(row: dict) -> tuple[str, list[str], int]:
    choice_data = row["choices"]
    choices = [str(text).strip() for text in choice_data["text"]]
    labels = [str(label) for label in choice_data["label"]]
    prompt = (
        "Answer the following multiple-choice science question.\n"
        f"Question: {str(row['question']).strip()}\n"
        "Answer:"
    )
    return prompt, choices, answer_index(row["answerKey"], labels)


def normalize_hellaswag(row: dict) -> tuple[str, list[str], int]:
    context = str(row.get("ctx") or f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}").strip()
    choices = [str(choice).strip() for choice in row["endings"]]
    prompt = (
        "Choose the most plausible continuation of the situation.\n"
        f"Context: {context}\n"
        "Continuation:"
    )
    return prompt, choices, int(row["label"])


def normalize_piqa(row: dict) -> tuple[str, list[str], int]:
    prompt = (
        "Choose the more sensible solution to the physical reasoning problem.\n"
        f"Goal: {str(row['goal']).strip()}\n"
        "Solution:"
    )
    return prompt, [str(row["sol1"]).strip(), str(row["sol2"]).strip()], int(row["label"])


def normalize_boolq(row: dict) -> tuple[str, list[str], int]:
    question = str(row["question"]).strip()
    if question and question[-1] not in "?!.":
        question += "?"
    prompt = (
        "Read the passage and answer the question with Yes or No.\n"
        f"Passage: {str(row['passage']).strip()}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt, ["No", "Yes"], int(bool(row["answer"]))


def normalize_winogrande(row: dict) -> tuple[str, list[str], int]:
    sentence = str(row["sentence"]).strip()
    options = [str(row["option1"]).strip(), str(row["option2"]).strip()]
    choices = [sentence.replace("_", option) for option in options]
    prompt = (
        "Choose the sentence with the correct resolution of the blank.\n"
        f"Sentence: {sentence}\n"
        "Resolved sentence:"
    )
    return prompt, choices, int(row["answer"]) - 1


def normalize_openbookqa(row: dict) -> tuple[str, list[str], int]:
    choice_data = row["choices"]
    choices = [str(text).strip() for text in choice_data["text"]]
    labels = [str(label) for label in choice_data["label"]]
    prompt = (
        "Answer the following multiple-choice science question.\n"
        f"Question: {str(row['question_stem']).strip()}\n"
        "Answer:"
    )
    return prompt, choices, answer_index(row["answerKey"], labels)


def normalize_truthfulqa(row: dict) -> tuple[str, list[str], int]:
    targets = row["mc1_targets"]
    choices = [str(choice).strip() for choice in targets["choices"]]
    labels = [int(label) for label in targets["labels"]]
    if 1 not in labels:
        raise ValueError("TruthfulQA MC1 row has no correct answer")
    prompt = (
        "Answer the question truthfully and avoid common misconceptions.\n"
        f"Question: {str(row['question']).strip()}\n"
        "Answer:"
    )
    return prompt, choices, labels.index(1)


NORMALIZERS = {
    "arc_easy": normalize_arc,
    "hellaswag": normalize_hellaswag,
    "piqa": normalize_piqa,
    "boolq": normalize_boolq,
    "winogrande": normalize_winogrande,
    "openbookqa": normalize_openbookqa,
    "truthfulqa_mc1": normalize_truthfulqa,
}


def selected_indices(size: int, limit: int, seed: int, task: str) -> list[int]:
    if limit <= 0 or limit >= size:
        return list(range(size))
    task_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(task))
    return sorted(random.Random(task_seed).sample(range(size), limit))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_task(task: str, output_dir: Path, limit: int, seed: int, force: bool) -> dict:
    spec = TASKS[task]
    output = output_dir / f"{task}.jsonl"
    if output.exists() and not force:
        rows = sum(1 for line in output.open("r", encoding="utf-8") if line.strip())
        return {
            "task": task,
            **spec,
            "rows": rows,
            "path": str(output),
            "sha256": sha256(output),
            "status": "existing",
        }

    args = [spec["dataset"]]
    if spec["config"]:
        args.append(spec["config"])
    dataset = load_dataset(*args, split=spec["split"])
    indices = selected_indices(len(dataset), limit, seed, task)
    normalizer = NORMALIZERS[task]

    temp = output.with_suffix(".jsonl.tmp")
    rows_written = 0
    with temp.open("w", encoding="utf-8") as handle:
        for source_index in indices:
            row = dict(dataset[int(source_index)])
            prompt, choices, label = normalizer(row)
            if not choices or not 0 <= label < len(choices):
                raise ValueError(f"Invalid normalized row for {task} index {source_index}")
            normalized = {
                "task": task,
                "source_index": int(source_index),
                "prompt": prompt,
                "choices": choices,
                "label": int(label),
            }
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            rows_written += 1
    temp.replace(output)
    return {
        "task": task,
        **spec,
        "source_rows": len(dataset),
        "rows": rows_written,
        "path": str(output),
        "sha256": sha256(output),
        "status": "prepared",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/benchmarks")
    parser.add_argument("--samples_per_task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop if any optional benchmark download fails.")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task in args.tasks:
        print(f"Preparing {task}...")
        try:
            record = prepare_task(task, output_dir, args.samples_per_task, args.seed, args.force)
            print(f"  {record['rows']} rows -> {record['path']}")
        except Exception as exc:
            if args.strict:
                raise
            record = {
                "task": task,
                **TASKS[task],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  skipped: {record['error']}")
        records.append(record)

    manifest = {
        "protocol": "deterministic zero-shot multiple-choice subsets",
        "samples_per_task": args.samples_per_task,
        "seed": args.seed,
        "tasks": records,
    }
    manifest_path = output_dir / "manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    with temp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    temp_manifest.replace(manifest_path)
    completed = sum(record.get("status") != "error" for record in records)
    if completed == 0:
        raise RuntimeError("No benchmark tasks were prepared")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
