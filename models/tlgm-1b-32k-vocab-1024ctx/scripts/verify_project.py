import argparse
import json
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/pretrain_tlgm_1b_32k_50b.yaml",
    "configs/sft_tlgm_1b_32k_chat.yaml",
    "configs/sft_tlgm_1b_32k_polish.yaml",
    "configs/sft_tlgm_1b_32k_reasoning.yaml",
]
EXPECTED_CHAIN = [
    (None, "checkpoints/tlgm_1b_32k_pretrain50b.pth"),
    ("checkpoints/tlgm_1b_32k_pretrain50b.pth", "checkpoints/tlgm_1b_32k_sft_chat.pth"),
    ("checkpoints/tlgm_1b_32k_sft_chat.pth", "checkpoints/tlgm_1b_32k_sft_polish.pth"),
    ("checkpoints/tlgm_1b_32k_sft_polish.pth", "checkpoints/tlgm_1b_32k_sft_reasoning.pth"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require_artifacts", action="store_true")
    args = parser.parse_args()

    configs = []
    for name in CONFIGS:
        with (PROJECT_DIR / name).open("r", encoding="utf-8") as handle:
            configs.append(yaml.safe_load(handle))
    reference_model = configs[0]["model"]
    errors = []
    for name, config, (expected_init, expected_output) in zip(CONFIGS, configs, EXPECTED_CHAIN):
        if config["model"] != reference_model:
            errors.append(f"{name}: model block differs from pretraining")
        if config["model"]["vocab_size"] != 32_000:
            errors.append(f"{name}: vocabulary is not 32,000")
        if config["model"]["context_length"] != 1_024:
            errors.append(f"{name}: context length is not 1,024")
        if config.get("init_checkpoint") != expected_init:
            errors.append(f"{name}: init checkpoint is {config.get('init_checkpoint')!r}, expected {expected_init!r}")
        if config["checkpoint_path"] != expected_output:
            errors.append(f"{name}: output checkpoint does not match the intended chain")
        if config["training"]["dtype"] != "bfloat16":
            errors.append(f"{name}: Blackwell training must use bfloat16")

    pretrain = configs[0]
    positions_per_step = (
        pretrain["training"]["batch_size"]
        * pretrain["training"]["gradient_accumulation_steps"]
        * pretrain["model"]["context_length"]
    )
    scheduled_positions = positions_per_step * pretrain["training"]["max_steps"]
    if scheduled_positions < 50_000_000_000 or scheduled_positions - 50_000_000_000 >= positions_per_step:
        errors.append(f"Pretraining schedule covers {scheduled_positions:,}, not one rounded 50B pass")
    if "init_checkpoint" in pretrain:
        errors.append("Pretraining config must not contain init_checkpoint")

    if args.require_artifacts:
        for relative in ("tokenizer/vocab.json", "tokenizer/merges.txt", pretrain["data"]["train_bin"], pretrain["data"]["meta"]):
            if not (PROJECT_DIR / relative).is_file():
                errors.append(f"Missing required artifact: {relative}")

    if errors:
        raise SystemExit("Project verification failed:\n- " + "\n- ".join(errors))
    print(
        json.dumps(
            {
                "status": "ok",
                "vocab_size": reference_model["vocab_size"],
                "context_length": reference_model["context_length"],
                "pretrain_positions_per_step": positions_per_step,
                "pretrain_steps": pretrain["training"]["max_steps"],
                "scheduled_pretrain_positions": scheduled_positions,
                "random_initialization": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
