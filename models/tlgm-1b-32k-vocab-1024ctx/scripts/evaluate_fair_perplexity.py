import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig  # noqa: E402
from tlgm.model import TLGMForCausalLM  # noqa: E402
from tlgm.tokenizer import load_tokenizer  # noqa: E402


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_hub_revision(repository: str, kind: str = "model") -> str:
    api = HfApi()
    info = api.model_info(repository) if kind == "model" else api.dataset_info(repository)
    if not info.sha:
        raise RuntimeError(f"Could not resolve immutable revision for {repository}")
    return info.sha


def load_corpus(spec: dict, cache_dir: Path) -> dict:
    revision = spec.get("revision") or resolve_hub_revision(spec["repository"], "dataset")
    dataset = load_dataset(
        spec["repository"],
        spec.get("config"),
        split=spec.get("split", "test"),
        revision=revision,
        cache_dir=str(cache_dir),
    )
    texts = [str(value) for value in dataset["text"]]
    text = "\n\n".join(texts)
    if not text.strip():
        raise ValueError(f"Dataset {spec['name']} produced no text")
    encoded = text.encode("utf-8")
    return {
        "name": spec["name"],
        "repository": spec["repository"],
        "config": spec.get("config"),
        "split": spec.get("split", "test"),
        "revision": revision,
        "rows": len(texts),
        "bytes": len(encoded),
        "sha256_utf8": hashlib.sha256(encoded).hexdigest(),
        "text": text,
    }


class ModelAdapter:
    def __init__(self, name, model, encode, bos_id, pad_id, device, dtype, metadata):
        self.name = name
        self.model = model
        self.encode = encode
        self.bos_id = int(bos_id)
        self.pad_id = int(pad_id)
        self.device = device
        self.dtype = dtype
        self.metadata = metadata


def load_tlgm(spec: dict, device: str, dtype: torch.dtype) -> ModelAdapter:
    config_path = resolve(spec["config"])
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = TLGMConfig.from_dict(yaml.safe_load(handle)["model"])
    checkpoint_path = resolve(spec["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TLGMForCausalLM(model_config)
    model.load_state_dict(payload["model"] if "model" in payload else payload, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = load_tokenizer(resolve(spec["tokenizer"]))
    metadata = {
        "type": "tlgm",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": payload.get("metadata", {}),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": model_config.vocab_size,
    }
    return ModelAdapter(
        spec["name"],
        model,
        lambda text: tokenizer.encode(text).ids,
        tokenizer.token_to_id("<bos>"),
        tokenizer.token_to_id("<pad>"),
        device,
        dtype,
        metadata,
    )


def load_huggingface(spec: dict, device: str, dtype: torch.dtype, cache_dir: Path) -> ModelAdapter:
    repository = spec["repository"]
    revision = spec.get("revision") or resolve_hub_revision(repository)
    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision, cache_dir=str(cache_dir))
    model = AutoModelForCausalLM.from_pretrained(
        repository,
        revision=revision,
        cache_dir=str(cache_dir),
        dtype=dtype,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = model.to(device).eval()
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    if bos_id is None:
        raise ValueError(f"{repository} has neither BOS nor EOS token")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else bos_id
    metadata = {
        "type": "huggingface",
        "repository": repository,
        "revision": revision,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": len(tokenizer),
    }
    return ModelAdapter(
        spec["name"],
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        bos_id,
        pad_id,
        device,
        dtype,
        metadata,
    )


def token_segments(token_ids: list[int], context_length: int) -> list[list[int]]:
    payload_length = context_length - 1
    if payload_length <= 0:
        raise ValueError("context_length must be at least 2")
    return [token_ids[start : start + payload_length] for start in range(0, len(token_ids), payload_length)]


@torch.inference_mode()
def score_corpus(adapter: ModelAdapter, corpus: dict, context_length: int, batch_size: int) -> dict:
    token_ids = adapter.encode(corpus["text"])
    if not token_ids:
        raise ValueError(f"Tokenizer for {adapter.name} produced no tokens")
    segments = token_segments(token_ids, context_length)
    total_nll = 0.0
    scored_tokens = 0
    if adapter.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in tqdm(range(0, len(segments), batch_size), desc=f"{adapter.name}/{corpus['name']}"):
        batch = segments[offset : offset + batch_size]
        width = max(len(segment) for segment in batch) + 1
        input_ids = torch.full((len(batch), width), adapter.pad_id, dtype=torch.long, device=adapter.device)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=adapter.device)
        labels = torch.full((len(batch), width - 1), -100, dtype=torch.long, device=adapter.device)
        for row_index, segment in enumerate(batch):
            sequence = [adapter.bos_id] + segment
            length = len(sequence)
            input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long, device=adapter.device)
            attention_mask[row_index, :length] = 1
            labels[row_index, : length - 1] = torch.tensor(segment, dtype=torch.long, device=adapter.device)
        autocast = torch.amp.autocast("cuda", dtype=adapter.dtype) if adapter.device.startswith("cuda") else torch.amp.autocast("cpu", enabled=False)
        with autocast:
            try:
                output = adapter.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except TypeError:
                output = adapter.model(input_ids)
            logits = output["logits"] if isinstance(output, dict) else output.logits
        shift_logits = logits[:, :-1, :].float().contiguous()
        total_nll += float(F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item())
        scored_tokens += sum(len(segment) for segment in batch)
    if adapter.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    nats_per_token = total_nll / scored_tokens
    bits_per_byte = total_nll / (corpus["bytes"] * math.log(2.0))
    return {
        "dataset": corpus["name"],
        "tokens": scored_tokens,
        "segments": len(segments),
        "bytes": corpus["bytes"],
        "total_negative_log_likelihood_nats": total_nll,
        "cross_entropy_nats_per_token": nats_per_token,
        "token_perplexity": math.exp(min(nats_per_token, 50.0)),
        "bits_per_byte": bits_per_byte,
        "byte_perplexity": 2.0**bits_per_byte,
        "tokens_per_byte": scored_tokens / corpus["bytes"],
        "elapsed_seconds": elapsed,
        "tokens_per_second": scored_tokens / elapsed,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3 if adapter.device.startswith("cuda") else None,
    }


def markdown_report(run: dict) -> str:
    lines = [
        "# Fair 1B Language-Model Comparison",
        "",
        f"Generated: `{run['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"- Common context length: `{run['protocol']['context_length']}`",
        f"- Precision: `{run['protocol']['dtype']}`",
        "- Dataset text and UTF-8 byte denominator are identical for every model.",
        "- Every non-special text token is scored once with teacher forcing.",
        "- Each non-overlapping context segment starts with the model's BOS token.",
        "- Token perplexity is tokenizer-dependent. Bits per byte (BPB) is the primary cross-tokenizer metric; lower is better.",
        "",
    ]
    for dataset in run["datasets"]:
        lines.extend([
            f"## {dataset['name']}",
            "",
            f"Dataset revision: `{dataset['revision']}`  ",
            f"UTF-8 bytes: `{dataset['bytes']:,}`  ",
            f"Text SHA-256: `{dataset['sha256_utf8']}`",
            "",
            "| Model | Params | Token PPL | Bits/byte | Byte PPL | Tokens/byte | tok/s | Peak VRAM GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        rows = []
        for model in run["models"]:
            result = next((item for item in model.get("results", []) if item["dataset"] == dataset["name"]), None)
            if result:
                rows.append((result["bits_per_byte"], model, result))
        for _, model, result in sorted(rows, key=lambda row: (row[0], row[1]["name"])):
            lines.append(
                f"| {model['name']} | {model['metadata']['parameters'] / 1e9:.3f}B | "
                f"{result['token_perplexity']:.3f} | **{result['bits_per_byte']:.4f}** | "
                f"{result['byte_perplexity']:.3f} | {result['tokens_per_byte']:.3f} | "
                f"{result['tokens_per_second']:,.0f} | {result['peak_cuda_memory_gib']:.2f} |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "Token perplexity should only be compared cautiously because vocabulary and token boundaries differ.",
        "BPB divides total negative log-likelihood by the identical number of source bytes and is the fairest metric in this report.",
        "Neither metric directly measures instruction following, factual accuracy, safety, or reasoning; use the practical benchmark report for those capabilities.",
        "",
    ])
    return "\n".join(lines)


def plot_report(run: dict, path: Path) -> None:
    datasets = run["datasets"]
    figure, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 7), squeeze=False)
    figure.patch.set_facecolor("#f4f5f1")
    colors = ["#14857c" if model["name"].startswith("TLGM") else "#e57734" for model in run["models"]]
    for axis, dataset in zip(axes[0], datasets):
        rows = []
        row_colors = []
        for model, color in zip(run["models"], colors):
            result = next((item for item in model.get("results", []) if item["dataset"] == dataset["name"]), None)
            if result:
                rows.append((model["name"], result["bits_per_byte"]))
                row_colors.append(color)
        order = sorted(range(len(rows)), key=lambda index: rows[index][1], reverse=True)
        labels = [rows[index][0] for index in order]
        values = [rows[index][1] for index in order]
        ordered_colors = [row_colors[index] for index in order]
        bars = axis.barh(labels, values, color=ordered_colors)
        axis.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
        axis.set_title(dataset["name"], fontsize=15, fontweight="bold", color="#19334a")
        axis.set_xlabel("Bits per UTF-8 byte (lower is better)")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.16)
    figure.suptitle("Fair 1B Language-Model Comparison", fontsize=22, fontweight="bold", color="#19334a")
    figure.text(0.5, 0.02, "TLGM models in teal; public reference models in orange. Common 1,024-token context.", ha="center", color="#647080")
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def validate_config(config: dict) -> None:
    protocol = config.get("protocol", {})
    if int(protocol.get("context_length", 0)) < 2:
        raise ValueError("protocol.context_length must be at least 2")
    if int(protocol.get("batch_size", 0)) <= 0:
        raise ValueError("protocol.batch_size must be positive")
    if protocol.get("dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("protocol.dtype must be bfloat16, float16, or float32")
    datasets = config.get("datasets", [])
    models = config.get("models", [])
    if not datasets or not models:
        raise ValueError("At least one dataset and model are required")
    for collection, name in ((datasets, "dataset"), (models, "model")):
        names = [item.get("name") for item in collection]
        if any(not value for value in names) or len(names) != len(set(names)):
            raise ValueError(f"Every {name} must have a unique non-empty name")
    for spec in models:
        if spec.get("type") == "tlgm":
            for key in ("checkpoint", "config", "tokenizer"):
                if not resolve(spec[key]).exists():
                    raise FileNotFoundError(f"Missing TLGM {key}: {resolve(spec[key])}")
        elif spec.get("type") != "huggingface":
            raise ValueError(f"Unsupported model type: {spec.get('type')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fair_perplexity_1b.yaml")
    parser.add_argument("--models", nargs="+", help="Optional exact model names to evaluate")
    parser.add_argument("--restart", action="store_true", help="Ignore resumable results and start a new report")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()
    with resolve(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    if args.validate_only:
        print(f"Valid comparison config: {len(config['models'])} models, {len(config['datasets'])} datasets")
        return
    protocol = config["protocol"]
    output_dir = resolve(protocol["output_dir"])
    cache_dir = resolve(protocol["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    if result_path.exists() and not args.restart:
        run = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        run = {
            "protocol": protocol,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            },
            "datasets": [],
            "models": [],
        }
    device = str(protocol.get("device", "cuda"))
    if device == "cuda":
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    dtype_name = str(protocol.get("dtype", "bfloat16"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    context_length = int(protocol["context_length"])
    batch_size = int(protocol["batch_size"])

    corpora = []
    for spec in config["datasets"]:
        corpus = load_corpus(spec, cache_dir)
        corpora.append(corpus)
        public = {key: value for key, value in corpus.items() if key != "text"}
        existing = next((item for item in run["datasets"] if item["name"] == corpus["name"]), None)
        if existing:
            existing.update(public)
        else:
            run["datasets"].append(public)
    atomic_json(result_path, run)

    selected = [spec for spec in config["models"] if not args.models or spec["name"] in args.models]
    if args.models and len(selected) != len(args.models):
        known = {spec["name"] for spec in config["models"]}
        raise ValueError(f"Unknown model names: {set(args.models) - known}")
    for spec in selected:
        model_result = next((item for item in run["models"] if item["name"] == spec["name"]), None)
        completed = {item["dataset"] for item in model_result.get("results", [])} if model_result else set()
        if len(completed) == len(corpora):
            print(f"Skipping completed model: {spec['name']}")
            continue
        print(f"Loading model: {spec['name']}", flush=True)
        adapter = load_tlgm(spec, device, dtype) if spec["type"] == "tlgm" else load_huggingface(spec, device, dtype, cache_dir)
        if model_result is None:
            model_result = {"name": spec["name"], "metadata": adapter.metadata, "results": []}
            run["models"].append(model_result)
        else:
            model_result["metadata"] = adapter.metadata
        for corpus in corpora:
            if corpus["name"] in completed:
                continue
            result = score_corpus(adapter, corpus, context_length, batch_size)
            model_result["results"].append(result)
            atomic_json(result_path, run)
            print(json.dumps(result, indent=2), flush=True)
        del adapter.model, adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run["generated_at"] = datetime.now().astimezone().isoformat()
    atomic_json(result_path, run)
    report = markdown_report(run)
    report_path = output_dir / "FAIR_PERPLEXITY_REPORT.md"
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    plot_report(run, output_dir / "fair_comparison.png")
    print(f"Wrote comparison report: {report_path}")


if __name__ == "__main__":
    main()
