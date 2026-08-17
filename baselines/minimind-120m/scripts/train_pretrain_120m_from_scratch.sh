#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/run_minimind_train.py --config configs/pretrain_120m_2b_4060ti.yaml "$@"
