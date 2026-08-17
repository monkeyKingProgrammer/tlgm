import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            json.loads(line)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", default="data/processed/pretrain_2b")
    parser.add_argument("--sft", default="data/processed/sft_chat_mix.jsonl")
    parser.add_argument("--repair", default="data/processed/sft_chat_repair.jsonl")
    args = parser.parse_args()

    pretrain_dir = resolve(args.pretrain_dir)
    if pretrain_dir.exists():
        files = sorted(pretrain_dir.glob("*.jsonl")) + sorted(pretrain_dir.glob("*.jsonl.gz"))
        print(f"Pretrain shards: {len(files)}")
        manifest = pretrain_dir / "manifest.json"
        if manifest.exists():
            print(manifest.read_text(encoding="utf-8"))
    else:
        print(f"Missing pretrain dir: {pretrain_dir}")

    for label, path in (("SFT", resolve(args.sft)), ("Repair", resolve(args.repair))):
        if path.exists():
            print(f"{label} rows: {count_jsonl(path)} ({path})")
        else:
            print(f"Missing {label}: {path}")


if __name__ == "__main__":
    main()
