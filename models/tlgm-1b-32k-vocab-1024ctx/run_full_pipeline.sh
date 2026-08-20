#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "$HOME/venvs/tlgm" ]; then
  # shellcheck disable=SC1091
  source "$HOME/venvs/tlgm/bin/activate"
fi

# Avoid interrupted Xet shard responses corrupting a resumable data build.
# The token progress file allows the builder to continue from its last flush.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

TARGET_TOKENS=50000000000
mkdir -p tokenizer data/raw data/processed data/benchmarks_full checkpoints outputs

echo "TLGM-1B-32K pipeline"
echo "Vocabulary: 32,000"
echo "Context length: 1,024"
echo "Pretraining target: ${TARGET_TOKENS} tokens"
echo "Pretraining initialization: random"

python3 scripts/verify_project.py

if [ ! -f tokenizer/manifest.json ]; then
  python3 scripts/train_tokenizer.py --vocab_size 32000
else
  echo "Found completed tokenizer manifest; reusing the exact 32K tokenizer."
fi

if [ ! -f data/processed/pretrain50b_32k_tokens_meta.json ]; then
  python3 scripts/prepare_pretrain50b_tokens.py --target_tokens "$TARGET_TOKENS"
else
  echo "Found completed 50B token metadata; validating before reuse."
fi

python3 scripts/validate_pretrain_data.py \
  --expected_tokens "$TARGET_TOKENS" \
  --expected_vocab_size 32000

# This configuration has no init_checkpoint. With no new-project checkpoint,
# the trainer explicitly reports random initialization. Existing checkpoints
# are resumed automatically after an interrupted session.
python3 scripts/train.py --config configs/pretrain_tlgm_1b_32k_50b.yaml

if [ ! -f data/processed/sft_chat_32k.manifest.json ]; then
  python3 scripts/prepare_sft_data.py
fi
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_chat.yaml

if [ ! -f data/processed/sft_polish_32k.jsonl ]; then
  python3 scripts/prepare_polish_sft.py \
    --output data/processed/sft_polish_32k.jsonl \
    --repeats 750
fi
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_polish.yaml

if [ ! -f data/processed/sft_reasoning_32k.manifest.json ]; then
  python3 scripts/prepare_reasoning_sft.py
fi
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_reasoning.yaml

python3 scripts/test_prompts.py \
  --config configs/sft_tlgm_1b_32k_reasoning.yaml \
  --checkpoint checkpoints/tlgm_1b_32k_sft_reasoning_best.pth \
  --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 180

python3 scripts/prepare_benchmarks.py \
  --output_dir data/benchmarks_full \
  --samples_per_task 0 \
  --strict
python3 scripts/evaluate_benchmarks.py \
  --config configs/benchmark_tlgm_1b_32k.yaml \
  --resume
python3 scripts/evaluate_fair_perplexity.py \
  --config configs/fair_perplexity_tlgm_1b_32k.yaml \
  --models TLGM-1B-32K

echo "Pipeline complete. Start chat with: ./run_chat.sh"
