import argparse
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.trainer import TLGMTrainer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain_tlgm_1b_32k_50b.yaml")
    parser.add_argument("--stop_after_seconds", type=float)
    parser.add_argument("--max_steps", type=int)
    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    TLGMTrainer(config, PROJECT_DIR).train(stop_after_seconds=args.stop_after_seconds)


if __name__ == "__main__":
    main()
