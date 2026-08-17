import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def read_metadata(checkpoint: Path) -> dict | None:
    sidecar = metadata_path(checkpoint)
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            return json.load(f)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, dict):
            return payload.get("metadata")
    return None


def fmt_num(value) -> str:
    if value is None:
        return "not available"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_float(value, digits: int = 4) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "not available"
    try:
        seconds = max(0, float(seconds))
    except (TypeError, ValueError):
        return "not available"
    days = int(seconds // 86400)
    seconds -= days * 86400
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def print_start_summary(checkpoint: Path, meta: dict | None, config: dict, target_max_steps: int, hours: float | None, steps: int | None) -> None:
    train_cfg = config["training"]
    tokens_per_step = int(train_cfg["batch_size"]) * int(train_cfg["gradient_accumulation_steps"]) * int(train_cfg["context_length"])
    current_step = int((meta or {}).get("step", 0))
    max_steps = int(train_cfg["max_steps"])
    total_target_tokens = max_steps * tokens_per_step
    trained_tokens = int((meta or {}).get("total_tokens_trained", current_step * tokens_per_step))
    remaining_steps = max(0, max_steps - current_step)
    remaining_tokens = max(0, total_target_tokens - trained_tokens)
    session_steps = max(0, target_max_steps - current_step)
    session_tokens = session_steps * tokens_per_step
    elapsed_seconds = (meta or {}).get("cumulative_gpu_time_seconds")
    seconds_per_step = None
    if current_step > 0 and elapsed_seconds is not None:
        seconds_per_step = float(elapsed_seconds) / current_step
    remaining_seconds = seconds_per_step * remaining_steps if seconds_per_step is not None else None
    session_seconds = seconds_per_step * session_steps if seconds_per_step is not None else None
    print("=" * 72)
    print("120M pretraining session")
    print("=" * 72)
    print(f"Checkpoint path: {checkpoint}")
    if checkpoint.exists():
        print("Checkpoint found: yes")
        print(f"Current step: {fmt_num((meta or {}).get('step'))}")
        print(f"Cumulative tokens trained: {fmt_num((meta or {}).get('total_tokens_trained'))}")
        print(f"Cumulative GPU/training time hours: {fmt_float((meta or {}).get('cumulative_gpu_time_hours'), 3)}")
        print(f"Last loss: {fmt_float((meta or {}).get('last_loss'), 4)}")
        print(f"Last eval loss: {fmt_float((meta or {}).get('last_eval_loss'), 4)}")
        print(f"Last LR: {fmt_float((meta or {}).get('last_lr'), 8)}")
        print(f"Average seconds per step so far: {fmt_float(seconds_per_step, 3)}")
    else:
        print("Checkpoint found: no")
        print("Start mode: random initialization")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Total target tokens: {total_target_tokens:,}")
    print(f"Remaining full-run steps: {remaining_steps:,}")
    print(f"Remaining full-run tokens: {remaining_tokens:,}")
    print(f"Estimated full-run remaining time: {fmt_duration(remaining_seconds)}")
    print(f"Requested extra steps: {steps if steps is not None else 'no step limit'}")
    print(f"Requested hours: {hours if hours is not None else 'no time limit'}")
    print(f"This session target max step: {target_max_steps:,}")
    print(f"This session target tokens: {session_tokens:,}")
    print(f"Estimated time for requested step limit: {fmt_duration(session_seconds)}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Special daily pretraining launcher with auto-resume summary and optional session limits.")
    parser.add_argument("--config", default="configs/pretrain_120m_2b_4060ti.yaml")
    parser.add_argument("--steps", type=int, help="Train this many additional optimizer steps, unless --hours is reached first.")
    parser.add_argument("--hours", type=float, help="Train roughly this many hours in this session, unless --steps is reached first.")
    parser.add_argument("--dry_run", action="store_true", help="Print what would run, but do not start training.")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER, help="Extra args passed to run_minimind_train.py after --.")
    args = parser.parse_args()

    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.hours is not None and args.hours <= 0:
        raise SystemExit("--hours must be positive")

    config_path = resolve(args.config)
    config = load_yaml(config_path)
    checkpoint = resolve(config["output_checkpoint"])
    meta = read_metadata(checkpoint)
    current_step = int((meta or {}).get("step", 0))
    config_max_steps = int(config["training"]["max_steps"])
    target_max_steps = config_max_steps
    if args.steps is not None:
        target_max_steps = min(config_max_steps, current_step + args.steps)

    print_start_summary(checkpoint, meta, config, target_max_steps, args.hours, args.steps)

    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "run_minimind_train.py"),
        "--config",
        str(config_path),
        "--max_steps",
        str(target_max_steps),
    ]
    if args.hours is not None:
        cmd.extend(["--stop_after_seconds", str(args.hours * 3600)])
    if args.trainer_args:
        extra = args.trainer_args[1:] if args.trainer_args and args.trainer_args[0] == "--" else args.trainer_args
        cmd.extend(extra)

    print("Command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd, cwd=PROJECT_DIR))


if __name__ == "__main__":
    main()
