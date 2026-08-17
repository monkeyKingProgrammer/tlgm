#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d /home/user/venvs/tlgm ]; then
  source /home/user/venvs/tlgm/bin/activate
fi

mkdir -p data/processed checkpoints outputs remote_logs
if [ ! -f data/processed/sft_reasoning_large_v2.manifest.json ]; then
  python3 scripts/prepare_reasoning_sft.py
fi
python3 scripts/verify_sft_artifacts.py
python3 scripts/train.py --config configs/sft_tlgm_1b_reasoning_3day.yaml
