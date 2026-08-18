from pathlib import Path

import yaml

from tlgm.config import TLGMConfig, estimate_params


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_NAMES = (
    "pretrain_tlgm_1b_32k_50b.yaml",
    "sft_tlgm_1b_32k_chat.yaml",
    "sft_tlgm_1b_32k_polish.yaml",
    "sft_tlgm_1b_32k_reasoning.yaml",
)


def load_config(name: str) -> dict:
    with (PROJECT_DIR / "configs" / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_all_training_stages_use_identical_32k_model():
    configs = [load_config(name) for name in CONFIG_NAMES]
    assert all(config["model"] == configs[0]["model"] for config in configs)
    assert configs[0]["model"]["vocab_size"] == 32_000
    assert configs[0]["model"]["context_length"] == 1_024


def test_parameter_count_and_tied_vocabulary_cost():
    config = TLGMConfig.from_dict(load_config(CONFIG_NAMES[0])["model"])
    report = estimate_params(config)
    assert report["total"] == 1_064_351_744
    assert report["token_embedding"] == 65_536_000
    assert report["output_head"] == 0


def test_pretraining_is_random_and_covers_50b_once():
    config = load_config(CONFIG_NAMES[0])
    assert "init_checkpoint" not in config
    tokens_per_step = (
        config["training"]["batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * config["model"]["context_length"]
    )
    scheduled = config["training"]["max_steps"] * tokens_per_step
    assert 50_000_000_000 <= scheduled < 50_000_000_000 + tokens_per_step
