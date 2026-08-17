# TLGM-120M 1024-Context Variant

Token-Local Global-Mixing Language Model.

This project implements the custom TLGM architecture from `../TLGM_120M_Codex_Project_Prompt.md`. It is intentionally separate from MiniMind and does not use Transformer blocks, self-attention, QKV projections, Mamba, RNNs, LSTMs, or GRUs.

This folder is a fresh 1024-context training variant. It does not reuse the old 256-context TLGM checkpoint because the position-mixing weights have different shapes. Training starts from random initialization, then runs masked SFT, repair SFT, and polish SFT.

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

This 1024-context model:

```text
vocab_size=8192
context_length=1024
model_dim=768
global_blocks=16
local_hidden_dim=3072
feature_hidden_dim=3072
parameter estimate: about 121.0M
```

The global block count is reduced from 22 to 16 to keep the 1024-context model near the original 120M parameter target.

## Setup

```powershell
cd C:\Users\ADMIN\minimind\tlgm_120m_1024ctx
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

This 1024-context folder reuses the already prepared TLGM pretraining binary by path:

```text
..\tlgm_120m\data\processed\pretrain_tokens.bin
..\tlgm_120m\data\processed\pretrain_tokens_meta.json
```

So you do not need to re-tokenize the 2B-token pretraining set unless you want a different tokenizer.

## One-Command Full Training

Double-click this file, or run it from Command Prompt:

```bat
cd C:\Users\ADMIN\minimind\tlgm_120m_1024ctx
run_full_pretrain_sft_1024ctx.bat
```

The BAT runs:

```text
pretrain from random init
-> masked SFT
-> repair masked SFT
-> prepare polish data
-> polish SFT
-> simple prompt test
```

Final checkpoint:

```text
checkpoints\tlgm_120m_1024ctx_sft_polished.pth
```

Chat after training:

```bat
cd C:\Users\ADMIN\minimind\tlgm_120m_1024ctx
python chat_cli.py --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 120
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

## Knowledge 3B Continued Pretraining

To improve factual/world knowledge, use continued pretraining on a new 3B-token educational/encyclopedic mix, then rerun SFT:

```bat
cd C:\Users\ADMIN\minimind\tlgm_120m_1024ctx
run_knowledge3b_pretrain_sft.bat
```

This creates a separate final checkpoint:

```text
checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth
```

It does not overwrite the original 1024-context checkpoints.

On Windows, this pipeline uses `num_workers: 0` in the training configs. This avoids multiprocessing `MemoryError` when spawning dataloader workers for large memory-mapped/tokenized datasets.

Knowledge data mix:

```text
40% HuggingFaceTB/smollm-corpus fineweb-edu-dedup
30% HuggingFaceTB/smollm-corpus cosmopedia-v2
30% wikimedia/wikipedia 20231101.en
```

Additional training budget:

```text
Knowledge continued pretraining: 3,000,041,472 tokens
Post-knowledge chatmix SFT:        327,680,000 tokens
Post-knowledge repair SFT:          98,304,000 tokens
Post-knowledge polish SFT:          19,660,800 tokens
Total additional seen tokens:    3,445,686,272 tokens
```

Chat with the knowledge-improved checkpoint after the BAT completes:

```bat
python chat_cli.py --config configs\sft_tlgm_120m_knowledge3b_polish.yaml --checkpoint checkpoints\tlgm_120m_1024ctx_knowledge3b_sft_final.pth --temperature 0.3 --top_p 0.8 --top_k 20 --max_new_tokens 160
```
