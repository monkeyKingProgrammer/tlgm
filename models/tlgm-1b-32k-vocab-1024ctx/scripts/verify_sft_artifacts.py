import argparse
import hashlib
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/sft_reasoning_large_v2.manifest.json")
    args = parser.parse_args()
    manifest_path = resolve(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    failures = []
    for name in ("train", "validation"):
        expected = manifest[name]
        path = Path(expected["path"])
        if not path.is_absolute():
            path = resolve(str(path))
        if not path.exists():
            failures.append(f"{name}: missing {path}")
            continue
        if path.stat().st_size != int(expected["bytes"]):
            failures.append(f"{name}: size mismatch")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            failures.append(f"{name}: SHA-256 mismatch")
        else:
            print(f"{name}: OK, {expected['examples']:,} examples, SHA-256 {actual_hash}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
