import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Features, Value, load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
MINIMIND_DIR = REPO_ROOT / "MiniMind"
sys.path.insert(0, str(MINIMIND_DIR))

from dataset.lm_dataset import SFTDataset  # noqa: E402
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def jsonl_files(path: Path) -> list[str]:
    if path.is_dir():
        files = sorted([*path.glob("*.jsonl"), *path.glob("*.jsonl.gz")])
        if not files:
            raise FileNotFoundError(f"No JSONL files found in {path}")
        return [str(p) for p in files]
    return [str(path)]


class ShardedPretrainDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        features = Features({"text": Value("string"), "source": Value("string")})
        self.samples = load_dataset("json", data_files=jsonl_files(data_path), split="train", features=features)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample["text"]), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


def load_config(path: Path, overrides: argparse.Namespace) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    training = cfg.setdefault("training", {})
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "context_length",
        "learning_rate",
        "max_steps",
        "warmup_steps",
        "save_steps",
        "eval_steps",
        "dtype",
        "device",
    ):
        value = getattr(overrides, key)
        if value is not None:
            training[key] = value
    for key in ("data_path", "init_checkpoint", "output_checkpoint", "resume_checkpoint", "log_path"):
        value = getattr(overrides, key)
        if value is not None:
            cfg[key] = value
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_model(cfg: dict, device: str) -> MiniMindForCausalLM:
    model_cfg = MiniMindConfig(**cfg["model"])
    model = MiniMindForCausalLM(model_cfg)
    print(f"Model params: {count_params(model):.2f}M")
    return model.to(device)


def model_state_half(model: torch.nn.Module) -> dict:
    return {k: v.detach().half().cpu() for k, v in model.state_dict().items()}


def checkpoint_metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def state_dict_from_checkpoint(payload):
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def save_checkpoint(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model_state_half(model)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def save_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(tmp, path)


def save_resume(path: Path, model, optimizer, scaler, global_step: int, metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    payload = {
        "model": model_state_half(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "metadata": metadata,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    save_metadata(checkpoint_metadata_path(path), metadata)


@torch.no_grad()
def evaluate(model, loader, device: str, eval_batches: int, autocast_ctx) -> float | None:
    if eval_batches <= 0:
        return None
    model.eval()
    losses = []
    for i, (input_ids, labels) in enumerate(loader):
        if i >= eval_batches:
            break
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with autocast_ctx:
            out = model(input_ids, labels=labels)
            loss = out.loss + out.aux_loss
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic pretrain resume from output_checkpoint.")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--stop_after_seconds", type=float, help="Stop after this many seconds in the current run, saving before exit.")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device")
    parser.add_argument("--data_path")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--output_checkpoint")
    parser.add_argument("--resume_checkpoint")
    parser.add_argument("--log_path")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config), args)
    train_cfg = cfg["training"]
    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        device = "cuda:0"
    seed_everything(int(train_cfg.get("seed", 42)))

    tokenizer = AutoTokenizer.from_pretrained(resolve_path(cfg["tokenizer_path"]))
    data_path = resolve_path(cfg["data_path"])
    stage = train_cfg["stage"]
    dataset = (
        ShardedPretrainDataset(data_path, tokenizer, max_length=int(train_cfg["context_length"]))
        if stage == "pretrain"
        else SFTDataset(str(data_path), tokenizer, max_length=int(train_cfg["context_length"]))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=("cuda" in device),
        drop_last=True,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=("cuda" in device),
        drop_last=False,
    )

    model = make_model(cfg, device)
    out_path = resolve_path(cfg["output_checkpoint"])
    resume_path = resolve_path(cfg.get("resume_checkpoint"))
    if stage == "pretrain":
        resume_path = out_path
    auto_resume = stage == "pretrain" and not args.no_auto_resume
    auto_resume_found = bool(auto_resume and resume_path and resume_path.exists())
    init_checkpoint = resolve_path(cfg.get("init_checkpoint"))
    if init_checkpoint:
        print(f"Loading init checkpoint: {init_checkpoint}")
        payload = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(state_dict_from_checkpoint(payload), strict=True)
    elif auto_resume_found:
        print(f"Pretrain checkpoint found and will be auto-resumed: {resume_path}")
    elif stage == "pretrain":
        print("Pretraining from random initialization: no checkpoint loaded.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg["dtype"] == "float16" and "cuda" in device))
    global_step = 0
    cumulative_gpu_time_seconds = 0.0
    last_metadata = {}
    should_resume = (args.resume or auto_resume) and resume_path and resume_path.exists()
    if should_resume:
        print(f"Resuming from: {resume_path}")
        resume = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict_from_checkpoint(resume), strict=True)
        if isinstance(resume, dict) and "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        if isinstance(resume, dict) and "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])
        if isinstance(resume, dict):
            global_step = int(resume.get("global_step", resume.get("metadata", {}).get("step", 0)))
            last_metadata = dict(resume.get("metadata") or {})
            cumulative_gpu_time_seconds = float(last_metadata.get("cumulative_gpu_time_seconds", 0.0))
    elif auto_resume and resume_path:
        print(f"No pretrain checkpoint found at {resume_path}. Starting from random initialization.")

    dtype = train_cfg["dtype"]
    amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else None
    log_path = resolve_path(cfg["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    accum = int(train_cfg["gradient_accumulation_steps"])
    max_steps = int(train_cfg["max_steps"])
    save_steps = int(train_cfg["save_steps"])
    eval_steps = int(train_cfg["eval_steps"])
    warmup_steps = int(train_cfg["warmup_steps"])
    base_lr = float(train_cfg["learning_rate"])

    model.train()
    iterator = iter(loader)
    if "cuda" in device:
        torch.cuda.synchronize()
    run_start = time.perf_counter()
    progress = tqdm(total=max_steps, initial=global_step, desc=stage)
    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        finite_micro_batches = 0
        for _ in range(accum):
            try:
                input_ids, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                input_ids, labels = next(iterator)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None and "cuda" in device else nullcontext()
            with ctx:
                out = model(input_ids, labels=labels)
                loss = (out.loss + out.aux_loss) / accum
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * accum
            finite_micro_batches += 1

        if finite_micro_batches == 0:
            print("Skipping optimizer step: all micro-batches had non-finite loss.")
            continue

        global_step += 1
        lr = cosine_lr(global_step, max_steps, base_lr, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()

        tokens_per_step = int(train_cfg["batch_size"]) * accum * int(train_cfg["context_length"])
        total_tokens_trained = global_step * tokens_per_step
        if "cuda" in device:
            torch.cuda.synchronize()
        elapsed_this_run = time.perf_counter() - run_start
        cumulative_time = cumulative_gpu_time_seconds + elapsed_this_run
        record = {
            "step": global_step,
            "loss": running_loss / finite_micro_batches,
            "lr": lr,
            "total_tokens_trained": total_tokens_trained,
            "cumulative_gpu_time_seconds": cumulative_time,
        }
        if eval_steps and global_step % eval_steps == 0:
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None and "cuda" in device else nullcontext()
            record["eval_loss"] = evaluate(model, eval_loader, device, int(train_cfg.get("eval_batches", 0)), ctx)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        progress.set_postfix(loss=f"{record['loss']:.4f}", lr=f"{lr:.2e}")
        progress.update(1)

        if global_step % save_steps == 0 or global_step == max_steps:
            metadata = {
                "step": global_step,
                "max_steps": max_steps,
                "stage": stage,
                "checkpoint": str(out_path),
                "data_path": str(data_path),
                "model": cfg["model"],
                "training": train_cfg,
                "last_loss": record["loss"],
                "last_eval_loss": record.get("eval_loss"),
                "last_lr": lr,
                "tokens_per_step": tokens_per_step,
                "total_tokens_trained": total_tokens_trained,
                "cumulative_gpu_time_seconds": cumulative_time,
                "cumulative_gpu_time_hours": cumulative_time / 3600,
                "device": device,
                "dtype": dtype,
                "optimizer": optimizer.__class__.__name__,
                "checkpoint_format": "training_state" if stage == "pretrain" else "model_state_dict",
                "note": "Pretrain overwrites one resumable checkpoint file. SFT checkpoints are model-only unless a separate resume path is configured.",
            }
            if stage == "pretrain":
                save_resume(out_path, model, optimizer, scaler, global_step, metadata)
            else:
                save_checkpoint(out_path, model)
                save_metadata(checkpoint_metadata_path(out_path), metadata)
                if resume_path:
                    save_resume(resume_path, model, optimizer, scaler, global_step, metadata)
            print(f"\nSaved checkpoint at step {global_step}: {out_path}")

        if args.stop_after_seconds and elapsed_this_run >= args.stop_after_seconds:
            metadata = {
                "step": global_step,
                "max_steps": max_steps,
                "stage": stage,
                "checkpoint": str(out_path),
                "data_path": str(data_path),
                "model": cfg["model"],
                "training": train_cfg,
                "last_loss": record["loss"],
                "last_eval_loss": record.get("eval_loss"),
                "last_lr": lr,
                "tokens_per_step": tokens_per_step,
                "total_tokens_trained": total_tokens_trained,
                "cumulative_gpu_time_seconds": cumulative_time,
                "cumulative_gpu_time_hours": cumulative_time / 3600,
                "device": device,
                "dtype": dtype,
                "optimizer": optimizer.__class__.__name__,
                "checkpoint_format": "training_state" if stage == "pretrain" else "model_state_dict",
                "stop_reason": "time_limit",
                "requested_stop_after_seconds": args.stop_after_seconds,
                "note": "Checkpoint saved because the requested session time limit was reached.",
            }
            if stage == "pretrain":
                save_resume(out_path, model, optimizer, scaler, global_step, metadata)
            else:
                save_checkpoint(out_path, model)
                save_metadata(checkpoint_metadata_path(out_path), metadata)
                if resume_path:
                    save_resume(resume_path, model, optimizer, scaler, global_step, metadata)
            print(f"\nTime limit reached after {elapsed_this_run / 3600:.3f} hours. Saved checkpoint at step {global_step}: {out_path}")
            break

    progress.close()


if __name__ == "__main__":
    main()
