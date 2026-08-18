import math
import unittest
from pathlib import Path

from scripts.evaluate_benchmarks import encode_candidate, evaluate_task, score_candidates
from scripts.prepare_benchmarks import (
    normalize_arc,
    normalize_boolq,
    normalize_openbookqa,
    normalize_truthfulqa,
    normalize_winogrande,
    selected_indices,
)
from tlgm.config import TLGMConfig
from tlgm.model import TLGMForCausalLM
from tlgm.tokenizer import load_tokenizer
from conftest import build_test_tokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]


class BenchmarkPreparationTests(unittest.TestCase):
    def test_normalizers_map_labels(self):
        arc = {
            "question": "What is water?",
            "choices": {"text": ["A gas", "A liquid"], "label": ["A", "B"]},
            "answerKey": "B",
        }
        self.assertEqual(normalize_arc(arc)[2], 1)

        boolq = {"passage": "The sky appears blue.", "question": "is the sky blue", "answer": True}
        self.assertEqual(normalize_boolq(boolq)[2], 1)

        winogrande = {
            "sentence": "The trophy does not fit because _ is large.",
            "option1": "the trophy",
            "option2": "the case",
            "answer": "1",
        }
        prompt, choices, label = normalize_winogrande(winogrande)
        self.assertIn("_", prompt)
        self.assertNotIn("_", choices[0])
        self.assertEqual(label, 0)

        openbook = {
            "question_stem": "What provides light?",
            "choices": {"text": ["The Sun", "A rock"], "label": ["A", "B"]},
            "answerKey": "A",
        }
        self.assertEqual(normalize_openbookqa(openbook)[2], 0)

        truthful = {
            "question": "Can humans breathe underwater unaided?",
            "mc1_targets": {
                "choices": ["No", "Yes"],
                "labels": [1, 0],
            },
        }
        self.assertEqual(normalize_truthfulqa(truthful)[2], 0)

    def test_subset_selection_is_deterministic(self):
        first = selected_indices(1000, 50, 2026, "arc_easy")
        second = selected_indices(1000, 50, 2026, "arc_easy")
        other = selected_indices(1000, 50, 2026, "piqa")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 50)
        self.assertEqual(first, sorted(first))


class BenchmarkScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(build_test_tokenizer())
        cls.model = TLGMForCausalLM(
            TLGMConfig(
                vocab_size=8192,
                context_length=64,
                embed_dim=32,
                model_dim=32,
                num_global_blocks=1,
                local_hidden_dim=64,
                feature_hidden_dim=64,
                dropout=0.0,
            )
        ).eval()

    def test_candidate_boundary_has_scoreable_tokens(self):
        encoded = encode_candidate(self.tokenizer, "Question: 2+2?\nAnswer:", "4", 64)
        self.assertLessEqual(len(encoded["input_ids"]), 64)
        self.assertGreater(encoded["target_tokens"], 0)
        self.assertFalse(encoded["target_mask"][0])

    def test_batched_scores_are_finite(self):
        candidates = [
            {"question_index": 0, "choice_index": 0, "prompt": "Answer:", "choice": "yes"},
            {"question_index": 0, "choice_index": 1, "prompt": "Answer:", "choice": "no"},
        ]
        results = score_candidates(
            self.model,
            self.tokenizer,
            candidates,
            batch_size=2,
            device="cpu",
            dtype=None,
            max_length=64,
        )
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(math.isfinite(result["log_likelihood"]))
            self.assertTrue(math.isfinite(result["mean_log_likelihood"]))
            self.assertGreater(result["target_tokens"], 0)

    def test_task_evaluation_returns_both_accuracy_metrics(self):
        rows = [
            {
                "source_index": 1,
                "prompt": "Question: 2+2?\nAnswer:",
                "choices": ["4", "5"],
                "label": 0,
            },
            {
                "source_index": 2,
                "prompt": "Question: Is water wet?\nAnswer:",
                "choices": ["No", "Yes"],
                "label": 1,
            },
        ]
        summary, predictions = evaluate_task(
            self.model,
            self.tokenizer,
            "synthetic",
            rows,
            batch_size=2,
            device="cpu",
            dtype=None,
            max_length=64,
        )
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["choices_scored"], 4)
        self.assertIn("accuracy", summary)
        self.assertIn("accuracy_normalized", summary)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
