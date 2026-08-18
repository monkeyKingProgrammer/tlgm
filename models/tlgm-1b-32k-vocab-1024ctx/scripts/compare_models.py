import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from tlgm.config import TLGMConfig, estimate_params


def main() -> None:
    cfg = TLGMConfig()
    print("TLGM default parameter estimate:")
    for key, value in estimate_params(cfg).items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
