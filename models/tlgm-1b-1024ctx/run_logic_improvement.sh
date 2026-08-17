#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d /home/user/venvs/tlgm ]; then
  source /home/user/venvs/tlgm/bin/activate
fi

mkdir -p data/processed checkpoints outputs remote_logs
if [ ! -f data/processed/sft_logic.jsonl ]; then
  python3 scripts/prepare_logic_sft.py --output data/processed/sft_logic.jsonl --repeats 2500
fi
python3 scripts/train.py --config configs/sft_tlgm_1b_logic.yaml
