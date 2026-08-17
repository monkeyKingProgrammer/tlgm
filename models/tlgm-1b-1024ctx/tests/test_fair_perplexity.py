import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_fair_perplexity", PROJECT_DIR / "scripts" / "evaluate_fair_perplexity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UniformModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.vocabulary_size = vocabulary_size

    def forward(self, input_ids, **_kwargs):
        shape = (*input_ids.shape, self.vocabulary_size)
        return {"logits": torch.zeros(shape, dtype=torch.float32, device=input_ids.device)}


class FairPerplexityTests(unittest.TestCase):
    def test_segments_reserve_one_position_for_bos(self):
        self.assertEqual(MODULE.token_segments(list(range(8)), 4), [[0, 1, 2], [3, 4, 5], [6, 7]])

    def test_uniform_model_has_vocabulary_sized_perplexity(self):
        adapter = MODULE.ModelAdapter(
            "uniform",
            UniformModel(4),
            lambda text: [1] * len(text),
            bos_id=0,
            pad_id=0,
            device="cpu",
            dtype=torch.float32,
            metadata={},
        )
        corpus = {"name": "tiny", "text": "abcd", "bytes": 4}
        result = MODULE.score_corpus(adapter, corpus, context_length=3, batch_size=2)
        self.assertAlmostEqual(result["token_perplexity"], 4.0, places=5)
        self.assertAlmostEqual(result["bits_per_byte"], 2.0, places=5)
        self.assertEqual(result["tokens"], 4)

    def test_config_validation_rejects_duplicate_names(self):
        config = {
            "protocol": {"context_length": 4, "batch_size": 1, "dtype": "float32"},
            "datasets": [
                {"name": "same", "repository": "one"},
                {"name": "same", "repository": "two"},
            ],
            "models": [{"name": "model", "type": "huggingface", "repository": "repo"}],
        }
        with self.assertRaises(ValueError):
            MODULE.validate_config(config)

    def test_markdown_report_handles_tied_scores(self):
        run = {
            "generated_at": "test",
            "protocol": {"context_length": 4, "dtype": "float32"},
            "datasets": [{"name": "tiny", "revision": "abc", "bytes": 4, "sha256_utf8": "def"}],
            "models": [
                {
                    "name": name,
                    "metadata": {"parameters": 4},
                    "results": [{
                        "dataset": "tiny",
                        "token_perplexity": 4.0,
                        "bits_per_byte": 2.0,
                        "byte_perplexity": 4.0,
                        "tokens_per_byte": 1.0,
                        "tokens_per_second": 10.0,
                        "peak_cuda_memory_gib": 0.0,
                    }],
                }
                for name in ("model-b", "model-a")
            ],
        }
        report = MODULE.markdown_report(run)
        self.assertIn("model-a", report)
        self.assertIn("model-b", report)


if __name__ == "__main__":
    unittest.main()
