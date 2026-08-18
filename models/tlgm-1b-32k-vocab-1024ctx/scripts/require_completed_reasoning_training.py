import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/tlgm_1b_32k_sft_reasoning.pth")
    parser.add_argument("--expected_step", type=int, default=120_000)
    args = parser.parse_args()
    checkpoint = resolve(args.checkpoint)
    metadata_path = checkpoint.with_suffix(".json")
    if not checkpoint.exists() or not metadata_path.exists():
        print(f"Final checkpoint or metadata is missing: {checkpoint}", file=sys.stderr)
        raise SystemExit(3)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    step = int(metadata.get("step", -1))
    max_steps = int(metadata.get("max_steps", args.expected_step))
    if step != args.expected_step or max_steps != args.expected_step:
        print(
            f"Training is incomplete: checkpoint step={step:,}, expected={args.expected_step:,}, max_steps={max_steps:,}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    print(f"Training completion verified at step {step:,}: {checkpoint}")


if __name__ == "__main__":
    main()
