import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_meta(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_tlgm_1b_20b.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ckpt = PROJECT_DIR / config["checkpoint_path"]
    meta = read_meta(ckpt)
    current_step = int(meta.get("step", 0))
    max_steps = int(config["training"]["max_steps"])
    target_step = min(max_steps, current_step + args.steps) if args.steps else max_steps
    tokens_per_step = int(config["training"]["batch_size"]) * int(config["training"]["gradient_accumulation_steps"]) * int(config["model"]["context_length"])
    total_tokens = max_steps * tokens_per_step
    trained_tokens = int(meta.get("total_tokens_trained", current_step * tokens_per_step))
    elapsed = float(meta.get("cumulative_gpu_time_seconds", 0.0))
    sec_per_step = elapsed / current_step if current_step else None
    remain_steps = max_steps - current_step
    remain_tokens = total_tokens - trained_tokens
    print("=" * 72)
    print("TLGM daily training session")
    print(f"Checkpoint: {ckpt}")
    print(f"Checkpoint found: {ckpt.exists()}")
    print(f"Current step: {current_step:,}")
    print(f"Cumulative tokens trained: {trained_tokens:,}")
    print(f"Remaining tokens: {remain_tokens:,}")
    print(f"Last loss: {meta.get('last_loss', 'not available')}")
    print(f"Last lr: {meta.get('last_lr', 'not available')}")
    print(f"Cumulative hours: {elapsed / 3600:.3f}")
    print(f"Estimated remaining hours: {(sec_per_step * remain_steps / 3600):.2f}" if sec_per_step else "Estimated remaining hours: not available")
    print(f"Session target step: {target_step:,}")
    print("=" * 72)
    cmd = [sys.executable, str(PROJECT_DIR / "scripts" / "train.py"), "--config", str(config_path), "--max_steps", str(target_step)]
    if args.hours:
        cmd.extend(["--stop_after_seconds", str(args.hours * 3600)])
    print("Command:", " ".join(cmd))
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd, cwd=PROJECT_DIR))


if __name__ == "__main__":
    main()
