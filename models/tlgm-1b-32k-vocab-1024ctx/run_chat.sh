#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "$HOME/venvs/tlgm" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venvs/tlgm/bin/activate"
fi

CHECKPOINT="checkpoints/tlgm_1b_32k_sft_reasoning_best.pth"
if [ ! -f "$CHECKPOINT" ]; then
  CHECKPOINT="checkpoints/tlgm_1b_32k_sft_reasoning.pth"
fi

python3 chat_cli.py \
  --config configs/sft_tlgm_1b_32k_reasoning.yaml \
  --checkpoint "$CHECKPOINT" \
  --dtype bfloat16 \
  --temperature 0.3 --top_p 0.8 --top_k 20 \
  --max_new_tokens 180 --history_turns 4
