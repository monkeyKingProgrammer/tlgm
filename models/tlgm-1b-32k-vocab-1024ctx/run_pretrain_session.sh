#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "$HOME/venvs/tlgm" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venvs/tlgm/bin/activate"
fi

HOURS="${1:-12}"
python3 scripts/verify_project.py --require_artifacts
python3 scripts/train_session.py \
  --config configs/pretrain_tlgm_1b_32k_50b.yaml \
  --hours "$HOURS"
