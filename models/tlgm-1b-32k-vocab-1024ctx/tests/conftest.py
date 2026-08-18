import tempfile
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import train_byte_bpe


_TOKENIZER_TEMP = tempfile.TemporaryDirectory()


def build_test_tokenizer() -> Path:
    root = Path(_TOKENIZER_TEMP.name)
    tokenizer_dir = root / "tokenizer"
    if not (tokenizer_dir / "vocab.json").exists():
        corpus = root / "corpus.txt"
        corpus.write_text(
            "User: Explain a small language model.\n"
            "Assistant: A language model predicts tokens from context.\n"
            "Reasoning mathematics science history stories answers.\n" * 100,
            encoding="utf-8",
        )
        train_byte_bpe([str(corpus)], tokenizer_dir, vocab_size=512, min_frequency=1)
    return tokenizer_dir
