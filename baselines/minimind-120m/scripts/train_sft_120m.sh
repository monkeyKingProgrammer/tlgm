#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/run_minimind_train.py --config configs/sft_120m_4060ti.yaml "$@"
