# TLGM: Token-Local Global-Mixing Language Models

TLGM is an experimental autoregressive language-model family that replaces
self-attention with learned causal lower-triangular sequence mixers. It does
not use Transformer attention, QKV projections, pretrained Transformer
weights, Mamba, LSTM, or GRU layers.

Every TLGM checkpoint in this project follows the same provenance:

```text
random initialization -> causal language pretraining -> supervised chat tuning
```

No Qwen, Llama, GPT, MiniMind, or other pretrained model checkpoint was used
to initialize TLGM. MiniMind appears only as a separately trained comparison
baseline and as the origin of some prepared public-data recipes.

This is a research and learning project, not a production ChatGPT replacement.
The models can generate English text and answer simple prompts, but they remain
weak in factual reliability, complex reasoning, coding, safety, and long
conversations.

## Model Family

| Track | Parameters | Vocabulary | Context | Pretraining | Status |
|---|---:|---:|---:|---:|---|
| TLGM-120M | ~118.1M | 8,192 | 256 | 2B tokens | Trained |
| TLGM-120M-1024 | ~121.0M | 8,192 | 1,024 | 2B + 3B continued tokens | Trained |
| TLGM-1B-1024 | 1,015,592,960 | 8,192 | 1,024 | 20B tokens | Trained |
| TLGM-1B-32K | 1,064,351,744 | 32,000 | 1,024 | 50B planned tokens | **Prepared, untrained** |
| MiniMind baseline | ~122M | 6,400 | 960 | 2B tokens | Comparison only |

The Git repository contains source, available tokenizer files, data builders, configs,
tests, launchers, loss histories, reports, and benchmark results. Large model
weights are distributed through versioned GitHub Releases rather than normal
Git history. See [WEIGHTS.md](WEIGHTS.md).

## Architecture

Input token IDs are converted to token and learned absolute-position
embeddings. A shared per-token encoder processes features before a stack of
global mixing blocks:

```text
token IDs
  -> token embedding + absolute position embedding
  -> shared position-wise encoder
  -> N causal global mixing blocks
  -> final normalization
  -> tied token-embedding output projection
  -> next-token logits
```

Each global block has two pre-normalized residual branches:

```text
X1 = X + TokenMix2(GELU(TokenMix1(LayerNorm(X))))
Y  = X1 + FeatureMLP(LayerNorm(X1))
```

For sequence length `N`, each causal token mixer uses a learned matrix shared
across feature channels:

```text
Y[b,n,d] = bias[n] + sum(A[n,j] * X[b,j,d], j=0..n)
```

The strict lower-triangular mask prevents future-token leakage. Unlike
attention, `A[n,j]` depends on source and destination positions rather than
the token content. This makes the design simple and inspectable, but removes
content-dependent routing.

The completed 8K-vocabulary 1B model uses:

```yaml
vocab_size: 8192
context_length: 1024
embed_dim: 2048
model_dim: 2048
num_global_blocks: 27
local_hidden_dim: 8192
feature_hidden_dim: 8192
tie_embeddings: true
```

Exact 1B parameter allocation:

| Component | Parameters |
|---|---:|
| Tied token embedding/output | 16,777,216 |
| Position embedding | 2,097,152 |
| Shared encoder | 33,568,768 |
| 27 global blocks | 963,145,728 |
| Final normalization | 4,096 |
| **Total** | **1,015,592,960** |

The prepared 32K-vocabulary track keeps the same context, width, and 27-block
architecture. Its larger tied token embedding adds 48,758,784 parameters for
an exact total of 1,064,351,744. It has no tokenizer, data, or checkpoint yet;
all are generated from scratch when its pipeline is started.

The complete equations, initialization, parameter derivation, training
semantics, and inference behavior are documented in
[ARCHITECTURE_AUDIT.md](reports/1b/ARCHITECTURE_AUDIT.md).

## Repository Layout

```text
models/tlgm-120m/              256-context architecture and pipeline
models/tlgm-120m-1024ctx/      1024-context 120M architecture and pipeline
models/tlgm-1b-1024ctx/        complete 1B architecture and pipeline
models/tlgm-1b-32k-vocab-1024ctx/ planned 32K-vocabulary, 50B-token pipeline
baselines/minimind-120m/       separately trained Transformer baseline tooling
reports/1b/                    paper, audits, benchmark plan, proposals
reports/benchmarks/            final practical and perplexity results
reports/comparisons/           TLGM versus MiniMind reports
reports/training_logs/         loss histories and training-progress plots
DATASETS.md                    corpus lineage and preparation
MODEL_CARD.md                  capabilities, metrics, and limitations
WEIGHTS.md                     checkpoint and release-asset policy
```

## Data

The trained TLGM tracks use an 8,192-entry byte-level BPE trained from scratch. Special
IDs are `<pad>=0`, `<bos>=1`, `<eos>=2`, and `<unk>=3`.

The TLGM-1B 20B-token pretraining stream contains:

| Source | Share | Tokens |
|---|---:|---:|
| SmolLM FineWeb-Edu deduplicated | 40% | 8B |
| SmolLM Cosmopedia v2 | 25% | 5B |
| English Wikimedia Wikipedia | 20% | 4B |
| FineWeb-Edu sample-10BT | 15% | 3B |

The data itself is not redistributed in this repository. Builders stream the
upstream datasets and create resumable `uint16` token binaries locally. Users
must review and comply with every upstream dataset license and terms. Full
lineage and preparation commands are in [DATASETS.md](DATASETS.md).

The untrained 32K track will train a new 32,000-entry byte-level BPE from a 4GB
mixed-source sample, then build a 50B-token corpus containing 35% deduplicated
FineWeb-Edu, 25% Cosmopedia v2, 17% general FineWeb, 15% OpenWebMath, and 8%
English Wikipedia. Its exact budget, data card, runtime estimate, commands, and
expected performance are documented in
[its project README](models/tlgm-1b-32k-vocab-1024ctx/README.md).

## Installation

Python 3.10-3.12 and a CUDA-enabled PyTorch build are recommended. Each model
track is self-contained:

```bash
cd models/tlgm-1b-1024ctx
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
cd models\tlgm-120m-1024ctx
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training From Random Initialization

### TLGM-120M

```powershell
cd models\tlgm-120m
python scripts\train_tokenizer.py --pretrain_dir PATH_TO_PRETRAIN_JSONL --tokenizer_dir tokenizer
python scripts\prepare_data.py --pretrain_dir PATH_TO_PRETRAIN_JSONL --tokenizer_dir tokenizer --target_tokens 2000000000
python scripts\train.py --config configs\pretrain_tlgm_120m_2b.yaml
python scripts\train.py --config configs\sft_tlgm_120m_masked.yaml
python scripts\train.py --config configs\sft_tlgm_120m_repair_masked.yaml
python scripts\train.py --config configs\sft_tlgm_120m_polish.yaml
```

### TLGM-120M, 1024 context

```bat
cd models\tlgm-120m-1024ctx
run_full_pretrain_sft_1024ctx.bat
```

For the additional 3B-token knowledge continuation:

```bat
run_knowledge3b_pretrain_sft.bat
```

### TLGM-1B, 1024 context

Prepare the 20B-token stream:

```bash
cd models/tlgm-1b-1024ctx
python3 scripts/prepare_pretrain20b_tokens.py
python3 scripts/validate_pretrain20b.py
```

Run the complete pretraining and post-training chain:

```bash
./run_tlgm_1b_full_pipeline.sh
```

The 1B trainer saves one rolling checkpoint every 500 optimizer steps and
automatically resumes model, optimizer, sampler, scaler, and random state.
Training starts from random initialization only when no checkpoint exists.

Reasoning continuation after the normal SFT chain:

```bash
./run_reasoning_improvement_3day.sh
```

## Chat

After downloading or producing a final checkpoint:

```bash
cd models/tlgm-1b-1024ctx
python3 chat_cli.py \
  --config configs/sft_tlgm_1b_reasoning_3day.yaml \
  --checkpoint checkpoints/tlgm_1b_1024ctx_sft_reasoning_3day.pth \
  --temperature 0.3 --top_p 0.8 --top_k 20 \
  --max_new_tokens 160 --history_turns 4
```

The generator keeps short history, truncates to the most recent 1,024 tokens,
and stops at EOS or chat stop markers. TLGM currently has no KV cache and
recomputes the retained context for every generated token.

## Completed 1B Training

The final reasoning stage completed successfully on an NVIDIA RTX PRO 6000
Blackwell Workstation Edition:

| Metric | Result |
|---|---:|
| Final step | 120,000 / 120,000 |
| Context positions in reasoning stage | 7,864,320,000 |
| Supervised assistant tokens | 2,677,269,185 |
| Cumulative GPU time | 88.68 hours |
| Final training loss | 1.1042 |
| Best validation loss | 1.0480 |
| Best validation perplexity | 2.8521 |

The base 20B pretraining run and earlier chat/repair/polish stages preceded
this reasoning continuation. SFT context positions must not be interpreted as
additional broad pretraining knowledge tokens.

## Benchmarks

The final reasoning checkpoint was evaluated on 20,110 zero-shot multiple
choice questions using conditional continuation log-likelihood:

| Task | Accuracy | Length-normalized accuracy |
|---|---:|---:|
| ARC-Easy | 59.26% | 48.95% |
| HellaSwag | 32.59% | 37.35% |
| PIQA | 66.49% | 66.38% |
| BoolQ | 53.24% | 53.24% |
| WinoGrande | 51.07% | 53.12% |
| OpenBookQA | 17.60% | 31.40% |
| TruthfulQA MC1 | 16.40% | 17.01% |
| **Weighted aggregate** | **42.33%** | **43.98%** |

This is a custom reproducible protocol, not necessarily identical to official
leaderboard prompting. Raw predictions and metadata are under
[reports/benchmarks](reports/benchmarks/).

The common-corpus BF16 perplexity comparison at 1,024 context produced:

| Model | Parameters | Token PPL | Bits/byte | Peak VRAM |
|---|---:|---:|---:|---:|
| SmolLM2-1.7B | 1.711B | 9.289 | 0.7565 | 5.07 GiB |
| Qwen2.5-1.5B | 1.544B | 10.555 | 0.7844 | 8.68 GiB |
| TinyLlama-1.1B-3T | 1.100B | 8.683 | 0.8213 | 3.28 GiB |
| OLMo-1B | 1.177B | 13.708 | 0.8412 | 4.12 GiB |
| **TLGM-1B reasoning** | **1.016B** | **18.406** | **1.1845** | **2.46 GiB** |
| TLGM-1B original SFT | 1.016B | 22.266 | 1.2619 | 2.46 GiB |

Bits per byte is preferred for cross-tokenizer comparison. The reasoning run
improved TLGM token perplexity by 17.3%, but TLGM remains substantially behind
established 1B-class models in language-modeling quality.

## Hardware Used

| Stage | Hardware | Notes |
|---|---|---|
| 120M experiments | RTX 4060 Ti 16GB, 32GB RAM | Windows, FP16, batch/accumulation tuned for 16GB |
| 1B pretraining/SFT | RTX PRO 6000 Blackwell 96GB | Linux, BF16 for late-run numerical stability |
| 1B reasoning stage | RTX PRO 6000 Blackwell 96GB | Approximately 34-35 GiB training VRAM |

The 1B model's raw BF16 weights require about 1.89 GiB. The controlled
teacher-forced benchmark peaked at 2.46 GiB for TLGM, but training with AdamW,
gradients, activations, and optimizer moments requires much more memory.

## Known Limitations

- Position routing is learned but not content-dependent.
- Context is fixed at 1,024 and does not naturally extrapolate.
- Sequence mixing has quadratic `O(N^2)` context cost.
- Autoregressive generation has no reusable state or KV cache.
- The 8,192-token vocabulary uses more tokens per English byte than the
  larger tokenizers used by OLMo, Qwen, SmolLM, and TinyLlama.
- Twenty billion pretraining tokens are 15-900 times fewer than major
  similarly sized public models.
- Knowledge, truthfulness, mathematics, coding, multilingual support, and
  safety alignment are limited.
- Outputs can be incorrect, contradictory, biased, or fabricated.

Do not use these models as authoritative sources or for high-stakes medical,
legal, financial, safety, or security decisions.

## License

Project-authored code and documentation are licensed under the
[Apache License 2.0](LICENSE). Upstream datasets, MiniMind components, and
third-party dependencies retain their own licenses and terms. The repository
license does not grant rights to redistribute upstream datasets.
