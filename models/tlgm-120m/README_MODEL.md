# TLGM-120M

Token-Local Global-Mixing Language Model.

This project implements the custom TLGM architecture from `../TLGM_120M_Codex_Project_Prompt.md`. It is intentionally separate from MiniMind and does not use Transformer blocks, self-attention, QKV projections, Mamba, RNNs, LSTMs, or GRUs.

## Architecture

Pipeline:

```text
token ids
-> byte-level BPE tokenizer trained from scratch
-> token embedding + position embedding
-> shared per-token encoder
-> causal global mixing network
-> tied vocabulary projection
-> next-token loss
```

The global mixer keeps tensors as `[B, N, D]`. Sequence mixing is done by learned lower-triangular matrices shared across channels:

```text
y[b, n, d] = sum_{j <= n} W[n, j] * x[b, j, d] + bias[n]
```

This gives global learned causal mixing without attention or QKV. Feature mixing is a per-position MLP:

```text
x = x + TokenMix(LayerNorm(x))
x = x + FeatureMLP(LayerNorm(x))
```

Default model:

```text
vocab_size=8192
context_length=256
model_dim=768
global_blocks=22
local_hidden_dim=3072
feature_hidden_dim=3072
parameter estimate: about 118.07M
```

## Setup

```powershell
cd C:\Users\ADMIN\minimind\tlgm_120m
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

This project uses the same raw 2B-token text source prepared for the MiniMind 120M project:

```text
..\minimind_120m_tiny_chatgpt\data\processed\pretrain_2b
```

TLGM has its own tokenizer, trained from scratch.

Train tokenizer:

```powershell
python scripts\train_tokenizer.py --pretrain_dir ..\minimind_120m_tiny_chatgpt\data\processed\pretrain_2b --tokenizer_dir tokenizer
```

Prepare pretraining token binary:

```powershell
python scripts\prepare_data.py --pretrain_dir ..\minimind_120m_tiny_chatgpt\data\processed\pretrain_2b --tokenizer_dir tokenizer --target_tokens 2000000000
```

Output:

```text
data\processed\pretrain_tokens.bin
data\processed\pretrain_tokens_meta.json
```

## Pretraining

Daily session launcher:

```powershell
python scripts\train_session.py --hours 6
```

Train for steps or time, whichever comes first:

```powershell
python scripts\train_session.py --steps 1000 --hours 6
```

Direct trainer:

```powershell
python scripts\train.py --config configs\pretrain_tlgm_120m_2b.yaml
```

Checkpoint:

```text
checkpoints\tlgm_120m_pretrain.pth
checkpoints\tlgm_120m_pretrain.json
```

The checkpoint is overwritten in place every 500 steps.

## SFT

Prepare SFT data after creating `sft_chat_mix.jsonl` in the MiniMind 120M folder:

```powershell
python scripts\prepare_sft_data.py --sft_jsonl ..\minimind_120m_tiny_chatgpt\data\processed\sft_chat_mix.jsonl --tokenizer_dir tokenizer
```

The original SFT pipeline is plain LM training on flattened chat text. If chat output is poor, use the masked SFT pipeline below instead. It trains loss only on assistant answers.

Masked broad SFT:

```powershell
python scripts\train.py --config configs\sft_tlgm_120m_masked.yaml
```

Masked repair SFT:

```powershell
python scripts\train.py --config configs\sft_tlgm_120m_repair_masked.yaml
```

Final masked SFT checkpoint:

```text
checkpoints\tlgm_120m_sft_final.pth
```

Legacy plain-LM SFT:

```powershell
python scripts\train.py --config configs\sft_tlgm_120m.yaml
```

SFT checkpoint:

```text
checkpoints\tlgm_120m_sft.pth
```

## Generation

From pretrain checkpoint:

```powershell
python scripts\generate.py --checkpoint checkpoints\tlgm_120m_pretrain.pth --prompt "Hello"
```

From SFT checkpoint:

```powershell
python scripts\generate.py --checkpoint checkpoints\tlgm_120m_sft.pth --prompt "User: hello`nAssistant:"
```

## Tests

```powershell
python -m pytest tests
```

Tests cover:

- Tensor shapes
- Forward loss
- Generation
- Causality
- Parameter count

## Limitations

This is a research architecture. It is not expected to match a Transformer of the same size without empirical tuning. The causal lower-triangular mixer is structurally causal and expressive, but it lacks content-dependent attention, so long-context behavior may differ substantially from standard LLMs.
