#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/prepare_pretrain_2b.py \
  --target_tokens 2000000000 \
  --output_dir data/processed/pretrain_2b
