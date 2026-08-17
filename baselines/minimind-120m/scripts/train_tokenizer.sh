#!/usr/bin/env bash
set -euo pipefail

cat <<'TXT'
This project reuses the MiniMind tokenizer from ../MiniMind/model.
No model weights are reused.

Reason:
- MiniMind SFTDataset expects MiniMind chat-template special tokens.
- Reusing a tokenizer is not loading pretrained model weights.
- Training a new tokenizer would require changing the chat template/data loader.
TXT
