import argparse
import gzip
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.tokenizer import train_byte_bpe  # noqa: E402


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def iter_jsonl_files(path: Path):
    if path.is_dir():
        yield from sorted(path.glob("*.jsonl"))
        yield from sorted(path.glob("*.jsonl.gz"))
    else:
        yield path


def extract_training_text(pretrain_dir: Path, output_text: Path, max_samples: int) -> None:
    output_text.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_text.open("w", encoding="utf-8") as out:
        for file in iter_jsonl_files(pretrain_dir):
            opener = gzip.open if file.suffix == ".gz" else open
            with opener(file, "rt", encoding="utf-8") as f:
                for line in tqdm(f, desc=file.name):
                    if not line.strip():
                        continue
                    text = json.loads(line).get("text", "")
                    if text:
                        out.write(text.replace("\n", " ") + "\n")
                        count += 1
                    if max_samples and count >= max_samples:
                        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", default="../minimind_120m_tiny_chatgpt/data/processed/pretrain_2b")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--temp_text", default="data/processed/tokenizer_train.txt")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=200000)
    args = parser.parse_args()

    temp_text = resolve(args.temp_text)
    extract_training_text(resolve(args.pretrain_dir), temp_text, args.max_samples)
    train_byte_bpe([str(temp_text)], resolve(args.tokenizer_dir), vocab_size=args.vocab_size)
    print(f"Saved tokenizer to {resolve(args.tokenizer_dir)}")


if __name__ == "__main__":
    main()
