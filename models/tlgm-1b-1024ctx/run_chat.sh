#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d /home/user/venvs/tlgm ]; then
  source /home/user/venvs/tlgm/bin/activate
fi

exec python3 chat_cli.py \
  --config configs/sft_tlgm_1b_polish.yaml \
  --checkpoint checkpoints/tlgm_1b_1024ctx_sft_final.pth \
  --temperature "${TEMPERATURE:-0.3}" \
  --top_p "${TOP_P:-0.8}" \
  --top_k "${TOP_K:-20}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-160}" \
  --history_turns "${HISTORY_TURNS:-4}"
