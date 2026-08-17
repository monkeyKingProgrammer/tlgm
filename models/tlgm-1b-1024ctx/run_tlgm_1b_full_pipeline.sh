#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "$HOME/venvs/tlgm" ]; then
  source "$HOME/venvs/tlgm/bin/activate"
fi

mkdir -p data/processed checkpoints outputs remote_logs

if [ ! -f data/processed/pretrain20b_tokens_meta.json ]; then
  RESUME_ARGS=()
  if [ -f data/processed/pretrain20b_tokens.bin ]; then
    if [ -f data/processed/pretrain20b_tokens_progress.json ]; then
      echo "Found tokenization progress state; the tokenizer will resume from it."
    elif [ -z "${PRETRAIN_RESUME_TOKENS:-}" ]; then
      echo "Found incomplete pretraining bin without completion metadata."
      echo "Set PRETRAIN_RESUME_TOKENS to its verified completed-token offset before restarting."
      exit 1
    else
      RESUME_ARGS=(--resume_tokens "$PRETRAIN_RESUME_TOKENS")
    fi
  fi
  python scripts/prepare_pretrain20b_tokens.py \
    --target_tokens 20000000000 \
    --output_bin data/processed/pretrain20b_tokens.bin \
    --meta data/processed/pretrain20b_tokens_meta.json \
    "${RESUME_ARGS[@]}"
else
  echo "Found completed pretraining metadata, skipping tokenization."
fi

python scripts/validate_pretrain_data.py \
  --bin data/processed/pretrain20b_tokens.bin \
  --meta data/processed/pretrain20b_tokens_meta.json \
  --expected_tokens 20000000000

python scripts/train.py --config configs/pretrain_tlgm_1b_20b.yaml
python scripts/train.py --config configs/sft_tlgm_1b_chatmix.yaml
python scripts/train.py --config configs/sft_tlgm_1b_repair.yaml

if [ ! -f data/processed/sft_polish.jsonl ]; then
  python scripts/prepare_polish_sft.py --output data/processed/sft_polish.jsonl --repeats 500
fi

python scripts/train.py --config configs/sft_tlgm_1b_polish.yaml
python scripts/test_prompts.py \
  --config configs/sft_tlgm_1b_polish.yaml \
  --checkpoint checkpoints/tlgm_1b_1024ctx_sft_final.pth \
  --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 180
