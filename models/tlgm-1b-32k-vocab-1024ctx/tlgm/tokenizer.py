from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def train_byte_bpe(files: list[str], output_dir: Path, vocab_size: int = 32000, min_frequency: int = 2) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS)
    tokenizer.save_model(str(output_dir))


def load_tokenizer(tokenizer_dir: Path) -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(str(tokenizer_dir / "vocab.json"), str(tokenizer_dir / "merges.txt"))
