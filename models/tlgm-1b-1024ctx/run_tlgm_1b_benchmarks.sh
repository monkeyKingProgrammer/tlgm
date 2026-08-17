#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "$HOME/venvs/tlgm" ]; then
  source "$HOME/venvs/tlgm/bin/activate"
fi

FINAL_CHECKPOINT="checkpoints/tlgm_1b_1024ctx_sft_final.pth"
FINAL_METADATA="checkpoints/tlgm_1b_1024ctx_sft_final.json"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

if [ ! -f "$FINAL_CHECKPOINT" ] || [ ! -f "$FINAL_METADATA" ]; then
  echo "Final SFT checkpoint is not available yet:"
  echo "  $FINAL_CHECKPOINT"
  exit 1
fi

python - "$FINAL_METADATA" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    metadata = json.load(handle)
step = int(metadata.get("step", 0))
max_steps = int(metadata.get("max_steps", 0))
if max_steps <= 0 or step < max_steps:
    raise SystemExit(
        f"Final SFT stage is incomplete: step {step:,} / {max_steps:,}. "
        "Run this benchmark after the training pipeline finishes."
    )
print(f"Final SFT checkpoint verified: step {step:,} / {max_steps:,}")
PY

mkdir -p data/benchmarks outputs/benchmarks/tlgm_1b_final remote_logs

python scripts/prepare_benchmarks.py \
  --output_dir data/benchmarks \
  --samples_per_task "$SAMPLES_PER_TASK" \
  --seed 2026

python scripts/evaluate_benchmarks.py \
  --config configs/benchmark_tlgm_1b.yaml \
  --batch_size "$EVAL_BATCH_SIZE" \
  --resume

echo
echo "Benchmark complete."
echo "Report: outputs/benchmarks/tlgm_1b_final/benchmark_results.md"
