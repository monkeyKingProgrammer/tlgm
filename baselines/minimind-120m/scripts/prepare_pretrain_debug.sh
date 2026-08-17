#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/prepare_pretrain_2b.py \
  --target_tokens 10000000 \
  --output_dir data/tiny_debug/pretrain_10m
