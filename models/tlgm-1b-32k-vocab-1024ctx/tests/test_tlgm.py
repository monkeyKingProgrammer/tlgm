import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from tlgm.config import TLGMConfig
from tlgm.dataset import (
    ChatSFTDataset,
    EvenHoldoutTrainDataset,
    ResumableRandomSampler,
    TokenBinDataset,
    even_holdout_indices,
)
from tlgm.model import TLGMForCausalLM
from tlgm.trainer import TLGMTrainer
from conftest import build_test_tokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]


def small_config(**overrides) -> TLGMConfig:
    values = {
        "vocab_size": 128,
        "context_length": 8,
        "embed_dim": 32,
        "model_dim": 32,
        "num_global_blocks": 2,
        "local_hidden_dim": 64,
        "feature_hidden_dim": 64,
        "dropout": 0.0,
        "tie_embeddings": True,
        "initializer_range": 0.02,
        "scale_residual_projections": True,
    }
    values.update(overrides)
    return TLGMConfig(**values)


class TLGMArchitectureTests(unittest.TestCase):
    def test_exact_32k_parameter_count(self):
        config = TLGMConfig(
            vocab_size=32000,
            context_length=1024,
            embed_dim=2048,
            model_dim=2048,
            num_global_blocks=27,
            local_hidden_dim=8192,
            feature_hidden_dim=8192,
            dropout=0.0,
            tie_embeddings=True,
            initializer_range=0.02,
            scale_residual_projections=True,
        )
        with torch.device("meta"):
            model = TLGMForCausalLM(config)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 1_064_351_744)

    def test_output_embedding_is_tied(self):
        model = TLGMForCausalLM(small_config())
        self.assertIs(model.lm_head.weight, model.embeddings.token_embedding.weight)

    def test_future_tokens_do_not_change_prefix_logits(self):
        torch.manual_seed(10)
        model = TLGMForCausalLM(small_config()).eval()
        first = torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]])
        second = first.clone()
        second[0, 5:] = torch.tensor([11, 12, 13])
        with torch.inference_mode():
            first_logits = model(first)["logits"]
            second_logits = model(second)["logits"]
        self.assertEqual(float((first_logits[:, :5] - second_logits[:, :5]).abs().max()), 0.0)

    def test_initial_loss_and_backward_are_finite(self):
        torch.manual_seed(11)
        model = TLGMForCausalLM(small_config())
        input_ids = torch.randint(0, 128, (2, 8))
        result = model(input_ids, input_ids.clone())
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertLess(float(result["loss"].detach()), 10.0)
        result["loss"].backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


class TLGMDataTests(unittest.TestCase):
    def test_pretraining_labels_are_not_pre_shifted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(32, dtype=np.uint16).tofile(path)
            dataset = TokenBinDataset(path, context_length=8)
            input_ids, labels = dataset[0]
            dataset.close()
        self.assertTrue(torch.equal(input_ids, torch.arange(8)))
        self.assertTrue(torch.equal(labels, input_ids))
        self.assertEqual(int(labels[1]), 1)

    def test_sampler_resume_continues_exactly(self):
        dataset = TensorDataset(torch.arange(29))
        sampler = ResumableRandomSampler(dataset, seed=123)
        iterator = iter(sampler)
        first = [next(iterator) for _ in range(7)]
        state = sampler.state_dict()
        expected_next = [next(iterator) for _ in range(9)]

        resumed = ResumableRandomSampler(dataset, seed=123)
        resumed.load_state_dict(state)
        resumed_iterator = iter(resumed)
        actual_next = [next(resumed_iterator) for _ in range(9)]

        self.assertEqual(actual_next, expected_next)
        self.assertEqual(len(set(first + actual_next)), len(first + actual_next))

    def test_even_holdout_spans_corpus_without_training_overlap(self):
        dataset = TensorDataset(torch.arange(100))
        holdout = even_holdout_indices(len(dataset), 10)
        training = EvenHoldoutTrainDataset(dataset, holdout)
        training_values = {int(training[index][0]) for index in range(len(training))}
        self.assertEqual(len(holdout), 10)
        self.assertLess(holdout[0], 10)
        self.assertGreater(holdout[-1], 90)
        self.assertFalse(training_values.intersection(holdout))
        self.assertEqual(training_values.union(holdout), set(range(100)))

    def test_long_sft_sample_keeps_assistant_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            row = {
                "conversations": [
                    {"role": "user", "content": "x " * 300},
                    {"role": "assistant", "content": "the retained answer"},
                ]
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ChatSFTDataset(path, build_test_tokenizer(), context_length=32)
            input_ids, labels = dataset[0]
        self.assertEqual(len(input_ids), 32)
        self.assertEqual(int(input_ids[0]), dataset.bos_id)
        self.assertGreater(int((labels != -100).sum()), 1)
        self.assertEqual(int(labels[-1]), dataset.eos_id)

    def test_overlength_answer_keeps_reasoning_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            row = {
                "conversations": [
                    {"role": "user", "content": "solve this"},
                    {"role": "assistant", "content": "BEGIN " + "middle " * 100 + "ENDMARK"},
                ]
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ChatSFTDataset(path, build_test_tokenizer(), context_length=32)
            _, labels = dataset[0]
            supervised = [int(token) for token in labels if int(token) != -100]
            decoded = dataset.tokenizer.decode(supervised)
        self.assertIn("BEGIN", decoded)
        self.assertIn("ENDMARK", decoded)
        self.assertEqual(supervised[-1], dataset.eos_id)

    def test_each_assistant_turn_is_a_separate_training_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            row = {
                "conversations": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": "second answer"},
                ]
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ChatSFTDataset(path, build_test_tokenizer(), context_length=64)
            first_labels = dataset[0][1]
            second_labels = dataset[1][1]
            first_text = dataset.tokenizer.decode([int(token) for token in first_labels if int(token) != -100])
            second_text = dataset.tokenizer.decode([int(token) for token in second_labels if int(token) != -100])
        self.assertEqual(len(dataset), 2)
        self.assertIn("first answer", first_text)
        self.assertNotIn("second answer", first_text)
        self.assertNotIn("first answer", second_text)
        self.assertIn("second answer", second_text)

    def test_explicit_target_index_prevents_duplicate_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            row = {
                "target_index": 3,
                "conversations": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ignored target"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "selected target"},
                ],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dataset = ChatSFTDataset(path, build_test_tokenizer(), context_length=64)
            _, labels = dataset[0]
            decoded = dataset.tokenizer.decode([int(token) for token in labels if int(token) != -100])
        self.assertEqual(len(dataset), 1)
        self.assertNotIn("ignored target", decoded)
        self.assertIn("selected target", decoded)


class TLGMTrainerTests(unittest.TestCase):
    def test_checkpoint_resume_restores_sampler_and_token_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            token_path = project / "tokens.bin"
            meta_path = project / "tokens.json"
            (np.arange(256, dtype=np.uint16) % 128).tofile(token_path)
            meta_path.write_text(
                json.dumps({"tokens": 256, "target_tokens": 256, "dtype": "uint16"}),
                encoding="utf-8",
            )
            config = {
                "model": small_config().to_dict(),
                "data": {"train_bin": str(token_path), "meta": str(meta_path)},
                "checkpoint_path": str(project / "checkpoint.pth"),
                "log_path": str(project / "loss.jsonl"),
                "tensorboard_dir": str(project / "tensorboard"),
                "validation": {
                    "enabled": True,
                    "holdout_samples": 4,
                    "eval_steps": 1,
                    "eval_batches": 1,
                },
                "training": {
                    "device": "cpu",
                    "dtype": "float32",
                    "batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "max_steps": 2,
                    "warmup_steps": 1,
                    "save_steps": 1,
                    "grad_clip": 1.0,
                    "num_workers": 0,
                    "seed": 44,
                },
            }
            TLGMTrainer(config, project).train()
            first = torch.load(project / "checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(first["checkpoint_schema_version"], 2)
            self.assertEqual(len(first["optimizer"]["param_groups"]), 2)
            self.assertEqual(first["optimizer"]["param_groups"][1]["weight_decay"], 0.0)
            self.assertTrue(all(not tensor.is_floating_point() or tensor.dtype == torch.float32 for tensor in first["model"].values()))
            self.assertTrue((project / "checkpoint.pth.previous").exists())
            self.assertIn("sampler_state", first)
            self.assertIn("rng_state", first)
            self.assertEqual(first["metadata"]["total_seen_tokens"], 64)
            self.assertEqual(first["metadata"]["total_supervised_tokens"], 56)

            config["training"]["max_steps"] = 3
            TLGMTrainer(config, project).train()
            resumed = torch.load(project / "checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["global_step"], 3)
            self.assertEqual(resumed["metadata"]["total_seen_tokens"], 96)
            self.assertEqual(resumed["metadata"]["total_supervised_tokens"], 84)
            self.assertEqual(resumed["sampler_state"]["offset"], 12)

    def test_corrupt_primary_checkpoint_recovers_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            token_path = project / "tokens.bin"
            meta_path = project / "tokens.json"
            (np.arange(128, dtype=np.uint16) % 128).tofile(token_path)
            meta_path.write_text(json.dumps({"tokens": 128, "target_tokens": 128}), encoding="utf-8")
            config = {
                "model": small_config().to_dict(),
                "data": {"train_bin": str(token_path), "meta": str(meta_path)},
                "checkpoint_path": str(project / "checkpoint.pth"),
                "log_path": str(project / "loss.jsonl"),
                "tensorboard_dir": str(project / "tensorboard"),
                "training": {
                    "device": "cpu",
                    "dtype": "float32",
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "max_steps": 1,
                    "warmup_steps": 0,
                    "save_steps": 1,
                    "grad_clip": 1.0,
                    "num_workers": 0,
                    "seed": 45,
                },
            }
            TLGMTrainer(config, project).train()
            shutil.copyfile(project / "checkpoint.pth", project / "checkpoint.pth.previous")
            (project / "checkpoint.pth").write_bytes(b"truncated")
            config["training"]["max_steps"] = 2
            TLGMTrainer(config, project).train()
            recovered = torch.load(project / "checkpoint.pth", map_location="cpu", weights_only=False)
            self.assertEqual(recovered["global_step"], 2)


if __name__ == "__main__":
    unittest.main()
