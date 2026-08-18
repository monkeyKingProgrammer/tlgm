import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig  # noqa: E402
from tlgm.model import TLGMForCausalLM  # noqa: E402
from tlgm.tokenizer import load_tokenizer  # noqa: E402


TASK_DESCRIPTIONS = {
    "arc_easy": "Grade-school science multiple choice",
    "hellaswag": "Commonsense event continuation",
    "piqa": "Physical commonsense reasoning",
    "boolq": "Passage-based yes/no reading comprehension",
    "winogrande": "Pronoun and commonsense resolution",
    "openbookqa": "Open-book elementary science multiple choice",
    "truthfulqa_mc1": "Truthfulness against common misconceptions",
}


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def encode_candidate(tokenizer, prompt: str, choice: str, max_length: int) -> dict:
    continuation = choice if choice[:1].isspace() else f" {choice}"
    prompt_ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt).ids
    full_ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(prompt + continuation).ids
    boundary = common_prefix_length(prompt_ids, full_ids)
    if boundary >= len(full_ids):
        raise ValueError("Choice produced no scoreable continuation tokens")

    dropped = max(0, len(full_ids) - max_length)
    if dropped:
        full_ids = full_ids[dropped:]
        boundary = max(0, boundary - dropped)
    target_mask = [index >= boundary for index in range(len(full_ids))]
    target_mask[0] = False
    if not any(target_mask[1:]):
        raise ValueError("Choice has no scoreable tokens after context truncation")
    return {
        "input_ids": full_ids,
        "target_mask": target_mask,
        "target_tokens": sum(target_mask),
    }


def chunks(values: list, size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def score_candidates(model, tokenizer, candidates: list[dict], batch_size: int, device: str, dtype, max_length: int):
    pad_id = tokenizer.token_to_id("<pad>")
    results = []
    model.eval()
    for batch in chunks(candidates, batch_size):
        encoded = [
            encode_candidate(tokenizer, item["prompt"], item["choice"], max_length)
            for item in batch
        ]
        width = max(len(item["input_ids"]) for item in encoded)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        target_mask = torch.zeros((len(batch), width), dtype=torch.bool)
        for row_index, item in enumerate(encoded):
            length = len(item["input_ids"])
            input_ids[row_index, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            target_mask[row_index, :length] = torch.tensor(item["target_mask"], dtype=torch.bool)
        input_ids = input_ids.to(device, non_blocking=True)
        target_mask = target_mask.to(device, non_blocking=True)

        amp_context = (
            torch.amp.autocast("cuda", dtype=dtype)
            if dtype is not None and "cuda" in device
            else torch.amp.autocast("cpu", enabled=False)
        )
        with amp_context:
            logits = model(input_ids)["logits"]

        shift_logits = logits[:, :-1, :].float()
        shift_targets = input_ids[:, 1:]
        shift_mask = target_mask[:, 1:]
        selected_logits = shift_logits.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs = selected_logits - torch.logsumexp(shift_logits, dim=-1)
        token_log_probs = token_log_probs * shift_mask
        sums = token_log_probs.sum(dim=-1)
        counts = shift_mask.sum(dim=-1)

        for item, score_sum, count in zip(batch, sums.tolist(), counts.tolist()):
            if count <= 0:
                raise RuntimeError("Candidate scoring produced zero target tokens")
            results.append(
                {
                    **item,
                    "log_likelihood": float(score_sum),
                    "mean_log_likelihood": float(score_sum / count),
                    "target_tokens": int(count),
                }
            )
    return results


def confidence_interval(accuracy: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    margin = 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / count)
    return max(0.0, accuracy - margin), min(1.0, accuracy + margin)


def evaluate_task(model, tokenizer, task: str, rows: list[dict], batch_size: int, device: str, dtype, max_length: int):
    candidates = []
    for question_index, row in enumerate(rows):
        for choice_index, choice in enumerate(row["choices"]):
            candidates.append(
                {
                    "question_index": question_index,
                    "choice_index": choice_index,
                    "prompt": row["prompt"],
                    "choice": str(choice),
                }
            )

    start = time.perf_counter()
    if "cuda" in device:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    scored = score_candidates(model, tokenizer, candidates, batch_size, device, dtype, max_length)
    if "cuda" in device:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    grouped = [[] for _ in rows]
    for candidate in scored:
        grouped[candidate["question_index"]].append(candidate)

    predictions = []
    correct_raw = 0
    correct_normalized = 0
    total_choice_tokens = 0
    total_choices = 0
    random_baseline_sum = 0.0
    for question_index, (row, choices) in enumerate(zip(rows, grouped)):
        choices.sort(key=lambda item: item["choice_index"])
        raw_prediction = max(choices, key=lambda item: item["log_likelihood"])["choice_index"]
        normalized_prediction = max(choices, key=lambda item: item["mean_log_likelihood"])["choice_index"]
        label = int(row["label"])
        correct_raw += int(raw_prediction == label)
        correct_normalized += int(normalized_prediction == label)
        total_choice_tokens += sum(item["target_tokens"] for item in choices)
        total_choices += len(choices)
        random_baseline_sum += 1.0 / len(choices)
        predictions.append(
            {
                "task": task,
                "source_index": row.get("source_index"),
                "question_index": question_index,
                "prompt": row["prompt"],
                "choices": row["choices"],
                "label": label,
                "prediction_raw": raw_prediction,
                "prediction_normalized": normalized_prediction,
                "correct_raw": raw_prediction == label,
                "correct_normalized": normalized_prediction == label,
                "scores": [
                    {
                        "choice_index": item["choice_index"],
                        "log_likelihood": item["log_likelihood"],
                        "mean_log_likelihood": item["mean_log_likelihood"],
                        "target_tokens": item["target_tokens"],
                    }
                    for item in choices
                ],
            }
        )

    count = len(rows)
    accuracy_raw = correct_raw / count
    accuracy_normalized = correct_normalized / count
    raw_low, raw_high = confidence_interval(accuracy_raw, count)
    norm_low, norm_high = confidence_interval(accuracy_normalized, count)
    summary = {
        "task": task,
        "description": TASK_DESCRIPTIONS.get(task, task),
        "questions": count,
        "choices_scored": total_choices,
        "accuracy": accuracy_raw,
        "accuracy_normalized": accuracy_normalized,
        "accuracy_95ci": [raw_low, raw_high],
        "accuracy_normalized_95ci": [norm_low, norm_high],
        "random_baseline": random_baseline_sum / count,
        "mean_target_tokens_per_choice": total_choice_tokens / total_choices,
        "elapsed_seconds": elapsed,
        "questions_per_second": count / elapsed,
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if "cuda" in device else None
        ),
    }
    return summary, predictions


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp.replace(path)


def write_predictions(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def markdown_report(metadata: dict, summaries: list[dict]) -> str:
    lines = [
        "# TLGM 1B Practical Benchmark Results",
        "",
        f"- Checkpoint: `{metadata['checkpoint']}`",
        f"- Evaluated: `{metadata['evaluated_at']}`",
        f"- Device: `{metadata['device_name']}`",
        f"- Precision: `{metadata['dtype']}`",
        f"- Evaluation batch size: `{metadata['eval_batch_size']}`",
        "",
        "This is a custom zero-shot conditional log-likelihood protocol. It is useful",
        "for comparisons made with this exact pipeline, but it is not automatically",
        "identical to lm-evaluation-harness or official leaderboard prompting.",
        "",
        "| Task | Questions | Accuracy | Accuracy norm | Random | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['task']} | {item['questions']:,} | "
            f"{100 * item['accuracy']:.2f}% | "
            f"{100 * item['accuracy_normalized']:.2f}% | "
            f"{100 * item['random_baseline']:.2f}% | "
            f"{item['elapsed_seconds']:.1f} |"
        )
    total_questions = sum(item["questions"] for item in summaries)
    weighted_raw = sum(item["accuracy"] * item["questions"] for item in summaries) / total_questions
    weighted_norm = (
        sum(item["accuracy_normalized"] * item["questions"] for item in summaries)
        / total_questions
    )
    macro_raw = sum(item["accuracy"] for item in summaries) / len(summaries)
    macro_norm = sum(item["accuracy_normalized"] for item in summaries) / len(summaries)
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Questions: `{total_questions:,}`",
            f"- Weighted accuracy: `{100 * weighted_raw:.2f}%`",
            f"- Weighted normalized accuracy: `{100 * weighted_norm:.2f}%`",
            f"- Macro task accuracy: `{100 * macro_raw:.2f}%`",
            f"- Macro normalized task accuracy: `{100 * macro_norm:.2f}%`",
            "",
            "Raw accuracy uses the sum of continuation log probabilities. Normalized",
            "accuracy divides each choice score by its number of continuation tokens.",
            "Per-example scores are stored in the corresponding prediction JSONL files.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_tlgm_1b.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with resolve(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = args.tasks or config["tasks"]
    checkpoint_path = resolve(args.checkpoint or config["checkpoint"])
    model_config_path = resolve(config["model_config"])
    tokenizer_dir = resolve(config["tokenizer_dir"])
    benchmark_dir = resolve(config["benchmark_dir"])
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or int(config["eval_batch_size"])
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda":
        device = "cuda:0"
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    dtype_name = str(config.get("dtype", "float16"))
    dtype = torch.float16 if dtype_name == "float16" else torch.bfloat16 if dtype_name == "bfloat16" else None
    max_length = int(config.get("max_length", 1024))
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with model_config_path.open("r", encoding="utf-8") as handle:
        model_yaml = yaml.safe_load(handle)
    model_config = TLGMConfig.from_dict(model_yaml["model"])
    if max_length > model_config.context_length:
        raise ValueError("Benchmark max_length exceeds model context length")
    tokenizer = load_tokenizer(tokenizer_dir)
    model = TLGMForCausalLM(model_config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    checkpoint_metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    metadata = {
        "protocol": "zero-shot conditional continuation log-likelihood",
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": checkpoint_metadata,
        "model_config": model_config.to_dict(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "tasks": tasks,
        "dtype": dtype_name,
        "eval_batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if "cuda" in device else platform.processor(),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    atomic_json(output_dir / "run_metadata.json", metadata)

    summaries = []
    for task in tasks:
        data_path = benchmark_dir / f"{task}.jsonl"
        if not data_path.exists():
            print(f"Skipping {task}: missing {data_path}")
            continue
        summary_path = output_dir / f"{task}_summary.json"
        predictions_path = output_dir / f"{task}_predictions.jsonl"
        if args.resume and summary_path.exists() and predictions_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            summaries.append(summary)
            print(f"Resumed existing result: {task}")
            continue

        rows = load_jsonl(data_path)
        if not rows:
            print(f"Skipping {task}: no rows")
            continue
        print(f"Evaluating {task}: {len(rows)} questions")
        summary, predictions = evaluate_task(
            model,
            tokenizer,
            task,
            rows,
            batch_size,
            device,
            dtype,
            max_length,
        )
        atomic_json(summary_path, summary)
        write_predictions(predictions_path, predictions)
        summaries.append(summary)
        print(
            f"  accuracy={100 * summary['accuracy']:.2f}% "
            f"normalized={100 * summary['accuracy_normalized']:.2f}% "
            f"time={summary['elapsed_seconds']:.1f}s"
        )

    if not summaries:
        raise RuntimeError("No benchmark tasks were evaluated")
    total_questions = sum(item["questions"] for item in summaries)
    aggregate = {
        "questions": total_questions,
        "tasks": len(summaries),
        "weighted_accuracy": (
            sum(item["accuracy"] * item["questions"] for item in summaries) / total_questions
        ),
        "weighted_accuracy_normalized": (
            sum(item["accuracy_normalized"] * item["questions"] for item in summaries)
            / total_questions
        ),
        "macro_accuracy": sum(item["accuracy"] for item in summaries) / len(summaries),
        "macro_accuracy_normalized": (
            sum(item["accuracy_normalized"] for item in summaries) / len(summaries)
        ),
        "task_results": summaries,
    }
    atomic_json(output_dir / "benchmark_results.json", {**metadata, "aggregate": aggregate})
    report = markdown_report(metadata, summaries)
    report_path = output_dir / "benchmark_results.md"
    temp_report = report_path.with_suffix(".md.tmp")
    temp_report.write_text(report, encoding="utf-8")
    temp_report.replace(report_path)
    print(f"Wrote benchmark report: {report_path}")


if __name__ == "__main__":
    main()
