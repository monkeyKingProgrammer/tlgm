# TLGM 1B 1024-Context Architecture Specification

## 1. Document status

This is the authoritative as-built specification for the current
`tlgm_1b_1024ctx` implementation. It describes the corrected source tree used
by the active 20-billion-token pretraining run.

The model contains exactly **1,015,592,960 unique trainable parameters**. It
uses a context window of 1,024 tokens and an 8,192-token byte-level BPE
vocabulary. It is initialized randomly and does not load Transformer,
MiniMind, Llama, Qwen, GPT, or other pretrained weights.

TLGM is not a Transformer. It contains:

- no self-attention;
- no Q, K, or V projections;
- no attention score matrix or attention softmax;
- no recurrent hidden state;
- no convolution;
- no KV cache;
- no mixture-of-experts routing.

Cross-token communication is performed by learned lower-triangular sequence
matrices. Per-token feature processing is performed by multilayer
perceptrons. The lower-triangular mask makes the complete network causal.

The previous report is retained as `ARCHITECTURE_AUDIT_PRE_FIX.md` for
historical traceability only. It is not part of the current specification.

## 2. Authoritative source map

| Responsibility | File |
|---|---|
| Model configuration and parameter estimator | `tlgm/config.py` |
| Complete causal language model | `tlgm/model.py` |
| Token and absolute position embeddings | `tlgm/embeddings.py` |
| Shared per-token input encoder | `tlgm/local_encoder.py` |
| Repeated global mixing blocks | `tlgm/global_mixer.py` |
| Lower-triangular sequence operation | `tlgm/causal_linear.py` |
| Pretraining and SFT datasets | `tlgm/dataset.py` |
| Training, validation, resume, and checkpoints | `tlgm/trainer.py` |
| LR schedule, seeding, and atomic saves | `tlgm/utils.py` |
| Autoregressive sampling | `tlgm/generation.py` |
| Byte-level BPE tokenizer | `tlgm/tokenizer.py` |
| Pretraining configuration | `configs/pretrain_tlgm_1b_20b.yaml` |
| SFT configurations | `configs/sft_tlgm_1b_*.yaml` |
| 20B token corpus builder | `scripts/prepare_pretrain20b_tokens.py` |
| Corpus integrity validator | `scripts/validate_pretrain_data.py` |
| Training entry point | `scripts/train.py` |
| Interactive inference | `chat_cli.py` |

The Python source and YAML configuration together define the model. A
checkpoint stores the model state, optimizer state, mixed-precision scaler,
sampler state, random-number-generator state, global step, configuration, and
training metadata.

## 3. Exact model configuration

The 1B configuration is:

```yaml
model:
  vocab_size: 8192
  context_length: 1024
  embed_dim: 2048
  model_dim: 2048
  num_global_blocks: 27
  local_hidden_dim: 8192
  feature_hidden_dim: 8192
  dropout: 0.0
  pad_token_id: 0
  bos_token_id: 1
  eos_token_id: 2
  unk_token_id: 3
  tie_embeddings: true
  initializer_range: 0.02
  scale_residual_projections: true
```

Symbols used throughout this document:

| Symbol | Value | Meaning |
|---|---:|---|
| `B` | runtime dependent | Microbatch size |
| `N` | 1 to 1024 | Current sequence length |
| `V` | 8192 | Vocabulary size |
| `D_e` | 2048 | Embedding width |
| `D` | 2048 | Model/feature width |
| `H` | 8192 | MLP hidden width |
| `L` | 27 | Number of global blocks |
| `C` | 1024 | Maximum context length |

The normal activation layout is `[B, N, D]`. Token IDs and labels use
`[B, N]`. Vocabulary logits use `[B, N, V]`.

## 4. End-to-end graph

For token IDs `T` with shape `[B, N]`, the model executes:

```text
T
 |
 +--> token embedding [V, D_e] --------+
 |                                     +--> add --> dropout
 +--> absolute position [C, D_e] ------+
                                             |
                                             v
                              shared per-token encoder
                                             |
                                             v
                           27 x global mixing block
                                             |
                                             v
                                      final LayerNorm
                                             |
                                             v
                          tied token embedding projection
                                             |
                                             v
                                    logits [B, N, V]
```

There is no input projection in this configuration because `D_e == D`.
`input_projection` is therefore `nn.Identity`. The implementation supports a
linear input projection for configurations where these widths differ, but it
has zero parameters in this model.

In equations:

```text
X_0 = Encoder(TokenEmbedding(T) + PositionEmbedding(0..N-1))
X_l = GlobalBlock_l(X_(l-1)), l = 1..27
Z   = FinalLayerNorm(X_27)
Logits = Z E_token^T
```

`E_token` is shared by the input embedding and output language-model head.

## 5. Tokenizer

The tokenizer is a Hugging Face `tokenizers.ByteLevelBPETokenizer` trained for
this project. Its vocabulary and merge rules are stored in:

```text
tokenizer/vocab.json
tokenizer/merges.txt
```

The configured vocabulary size is 8,192. Byte-level BPE ensures arbitrary
input text can be represented without a word-level out-of-vocabulary failure.
The four special tokens are registered in this exact order:

| ID | Token | Purpose |
|---:|---|---|
| 0 | `<pad>` | SFT sequence padding |
| 1 | `<bos>` | Beginning of a document or chat prompt |
| 2 | `<eos>` | End of a document or generated response |
| 3 | `<unk>` | Unknown-token fallback |

The 20B corpus inserts `<bos>` before each source document and `<eos>` after
it. SFT samples begin with `<bos>` and end with `<eos>`. During generation,
sampling stops when token ID 2 is produced.

## 6. Embedding stage

`TokenPositionEmbeddings` owns:

```text
token_embedding:    Embedding(8192, 2048)
position_embedding: Embedding(1024, 2048)
dropout:             Dropout(0.0)
```

For batch item `b`, position `n`, and feature `d`:

```text
X_embed[b,n,d] =
    E_token[T[b,n],d] + E_position[n,d]
```

Absolute position IDs are generated as `0, 1, ..., N-1` on the same device as
the input. Prompt lengths greater than 1,024 are not accepted by the model.
Generation retains only the most recent 1,024 IDs before each forward pass.

The embedding stage contributes:

```text
8192 * 2048 = 16,777,216 token parameters
1024 * 2048 =  2,097,152 position parameters
```

## 7. Shared input encoder

Before the 27 global blocks, every token independently passes through one
shared feature encoder:

```text
Linear(2048, 8192)
GELU
Dropout(0.0)
Linear(8192, 2048)
LayerNorm(2048)
```

For each position:

```text
X_enc = LayerNorm(W_2 GELU(W_1 X_embed + b_1) + b_2)
```

The same weights are used at all positions. This stage cannot itself move
information between tokens; its purpose is nonlinear feature expansion and
compression before global sequence processing.

Parameter count:

```text
first linear weight       2048 * 8192 = 16,777,216
first linear bias                         8,192
second linear weight      8192 * 2048 = 16,777,216
second linear bias                        2,048
LayerNorm scale and bias   2 * 2048 =      4,096
total                                 33,568,768
```

## 8. Global mixing block

The model contains 27 distinct `GlobalMixingBlock` instances. Parameters are
not shared between blocks.

Each block has two pre-normalized residual branches:

```text
X -------------------------------+-------------------------------+
 |                               |                               |
 v                               |                               |
LayerNorm                        |                               |
 |                               |                               |
CausalSequenceLinear #1          |                               |
 |                               |                               |
GELU                             |                               |
 |                               |                               |
CausalSequenceLinear #2          |                               |
 |                               |                               |
 +------------------------------> add = X_token                  |
                                  |                              |
                                  v                              |
                               LayerNorm                         |
                                  |                              |
                             Linear D -> H                       |
                                  |                              |
                                GELU                             |
                                  |                              |
                              Dropout(0)                         |
                                  |                              |
                             Linear H -> D                       |
                                  |                              |
                              Dropout(0)                         |
                                  |                              |
                                  +----------------------------> add
                                                                 |
                                                                 v
                                                              output
```

For block input `X`:

```text
U = TokenNorm(X)
M = TokenMix2(GELU(TokenMix1(U)))
X' = X + M

F = W_4 GELU(W_3 FeatureNorm(X') + b_3) + b_4
Y = X' + F
```

Dropout is present structurally but is inactive because the configured rate
is zero.

### 8.1 Causal sequence mixing

Each `CausalSequenceLinear(C=1024)` contains:

```text
weight: [1024, 1024]
bias:   [1024]
mask:   lower-triangular [1024, 1024], non-persistent buffer
```

For a runtime sequence of length `N <= 1024`, the implementation slices the
top-left `N x N` portion and applies the causal mask:

```text
A = weight[0:N,0:N] * tril(ones(N,N))
Y[b,n,d] = bias[n] + sum(j=0..n) A[n,j] X[b,j,d]
```

The matrix is shared across batch items and all 2,048 feature channels. Each
output position may consume its own position and earlier positions, but never
a later one. Two such operators occur in every block.

The mask is registered with `persistent=False`; checkpoints store the learned
weights and biases but do not waste space storing a deterministic 1,024 x
1,024 mask for every operator. The mask is recreated when the model object is
constructed.

Unlike attention, sequence mixing weights depend on absolute source and
destination positions, not on token content. Content-dependent computation
still occurs in GELU and feature MLPs around the sequence operators.

### 8.2 Feature mixing

The feature branch is a conventional position-wise MLP:

```text
LayerNorm(2048)
Linear(2048, 8192)
GELU
Dropout(0.0)
Linear(8192, 2048)
Dropout(0.0)
```

It uses independent learned weights in each of the 27 blocks. It processes
each position independently after that position has received causal context
from the token-mixing branch.

### 8.3 Parameters per block

Two causal sequence operators:

```text
2 * (1024 * 1024 + 1024) = 2,099,200
```

Feature MLP:

```text
2048 * 8192 + 8192 + 8192 * 2048 + 2048 = 33,564,672
```

Two LayerNorms:

```text
2 norms * 2 tensors * 2048 = 8,192
```

Total:

```text
2,099,200 + 33,564,672 + 8,192 = 35,672,064 per block
35,672,064 * 27 = 963,145,728 across all blocks
```

The feature MLPs hold most of the parameters. Sequence matrices account for
about 56.7M parameters across the full network.

## 9. Final normalization and output head

After block 27:

```text
Z = LayerNorm(X_27)
Logits[b,n,v] = sum(d) Z[b,n,d] E_token[v,d]
```

The final LayerNorm has 4,096 parameters. The output head has no independent
parameter allocation because its weight object is tied to
`embeddings.token_embedding.weight`. Weight tying:

- reduces unique parameter count by 16,777,216;
- keeps input and output token representations in one shared matrix;
- causes updates from both embedding lookup and output prediction to modify
  the same parameter.

The output has shape `[B, N, 8192]`.

## 10. Causal language-model objective

Pretraining dataset windows contain exactly 1,024 consecutive tokens:

```text
input_ids = [t_0, t_1, ..., t_1023]
labels    = input_ids.clone()
```

The model performs the causal shift once:

```python
cross_entropy(
    logits[:, :-1].reshape(-1, vocab_size),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

The prediction pairs are therefore:

```text
position 0 predicts t_1
position 1 predicts t_2
...
position 1022 predicts t_1023
```

Each full pretraining sample contributes 1,023 supervised next-token targets.
The last logit has no target inside the same window.

SFT uses the same model-level shift. User text, role prefixes, padding, and
other non-answer locations are labeled `-100`, so they provide context without
contributing to cross entropy. Assistant answer tokens and the final `<eos>`
token are supervised.

## 11. Initialization

The model is initialized entirely from random values.

General initialization:

```text
Linear weights:       Normal(mean=0, std=0.02)
Embedding weights:    Normal(mean=0, std=0.02)
Linear biases:        zero
LayerNorm scales:     one
LayerNorm biases:     zero
```

Residual output projections receive additional depth-aware scaling.

For each global block:

```text
token_mix_1.weight:
    Xavier uniform, gain = 1

token_mix_2.weight:
    Xavier uniform, gain = 1 / sqrt(2L)

feature_mlp output weight:
    Normal(0, 0.02 / sqrt(2L))
```

With `L=27`, the residual scale denominator is:

```text
sqrt(54) ~= 7.34847
```

This keeps residual contributions appropriately scaled across the 27-block
depth. The tied token/output matrix remains one shared tensor and its final
initialized distribution follows the configured standard deviation of 0.02.

The full-size initialization check verifies:

- exact parameter count;
- finite forward loss;
- embedding standard deviation close to 0.02;
- successful CUDA construction and forward execution.

## 12. Exact parameter accounting

| Component | Unique parameters |
|---|---:|
| Token embedding / tied output matrix | 16,777,216 |
| Absolute position embedding | 2,097,152 |
| Input projection | 0 |
| Shared input encoder | 33,568,768 |
| 27 global blocks | 963,145,728 |
| Final LayerNorm | 4,096 |
| Independent output head | 0 |
| **Total** | **1,015,592,960** |

Raw model-weight storage, excluding optimizer and checkpoint metadata:

| Representation | Approximate size |
|---|---:|
| FP32 | 3,874.18 MiB |
| FP16 | 1,937.09 MiB |
| BF16 | 1,937.09 MiB |

Training requires substantially more GPU memory than model weights alone due
to gradients, AdamW moments, activations, temporary tensors, and the
mixed-precision scaler. The current RTX PRO 6000 run uses approximately
34.5 GiB at `batch_size=8`, `context_length=1024`, and FP16.

## 13. Pretraining corpus

The pretraining stream contains exactly 20,000,000,000 `uint16` token IDs.
The binary file is:

```text
data/processed/pretrain20b_tokens.bin
```

Its expected size is exactly:

```text
20,000,000,000 tokens * 2 bytes = 40,000,000,000 bytes
```

Corpus composition:

| Source | Weight | Token quota |
|---|---:|---:|
| SmolLM FineWeb-Edu deduplicated | 40% | 8.0B |
| SmolLM Cosmopedia v2 | 25% | 5.0B |
| English Wikimedia Wikipedia | 20% | 4.0B |
| FineWeb-Edu sample-10BT | 15% | 3.0B |
| **Total** | **100%** | **20.0B** |

The builder streams source rows from Hugging Face, rejects very short text,
tokenizes each document with the project tokenizer, adds document boundary
tokens, and writes directly into a preallocated memory-mapped `uint16` array.
Progress metadata permits interrupted data preparation to resume.

Before model construction, training checks:

- completion metadata exists;
- `tokens` is positive;
- `tokens == target_tokens`;
- binary byte size equals `tokens * sizeof(uint16)`.

The standalone validator also verifies source totals and metadata consistency.

## 14. Pretraining sampling

`TokenBinDataset` memory-maps the binary corpus and divides it into
non-overlapping 1,024-token windows. It does not load 40 GB into RAM.

The training subset excludes the final 10,000 windows, which form a fixed
held-out validation split.

Training uses `ResumableRandomSampler`. Instead of allocating an in-memory
random permutation with roughly 19.5 million Python integers, it generates a
bijective affine permutation:

```text
index(position) = (a * position + b) mod dataset_size
```

`a` is chosen coprime to the dataset size. Therefore each sample occurs once
per sampler epoch. A seed-derived pair `(a,b)` changes by epoch. The sampler
stores only:

```text
seed
epoch
offset
```

This gives deterministic randomized traversal with constant sampler memory
and exact checkpoint continuation.

## 15. Pretraining configuration

The active pretraining settings are:

```yaml
training:
  device: cuda
  dtype: float16
  batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 0.00015
  weight_decay: 0.1
  max_steps: 305176
  warmup_steps: 3000
  save_steps: 500
  grad_clip: 1.0
  num_workers: 0
  seed: 1001
```

The microbatch contains:

```text
8 sequences * 1024 tokens = 8,192 seen tokens
```

One optimizer step accumulates eight microbatches:

```text
8 * 8 * 1024 = 65,536 seen tokens per optimizer step
```

The complete schedule processes:

```text
305,176 * 65,536 = 20,000,014,336 seen tokens
```

This is approximately one pass over the 20B token stream. The small excess
comes from rounding the optimizer-step count upward to a whole accumulated
batch.

## 16. Optimizer, schedule, and numerical behavior

The optimizer is AdamW over all trainable parameters:

```text
base learning rate: 1.5e-4
weight decay:       0.1
gradient clipping:  global norm 1.0
```

The learning-rate schedule is linear warmup followed by cosine decay. For
step `s`, warmup `W`, maximum step `S`, and base rate `eta`:

```text
if s < W:
    lr = eta * s / W
else:
    p = (s - W) / (S - W)
    lr = 0.1 * eta + 0.9 * eta * 0.5 * (1 + cos(pi * p))
```

The schedule therefore warms from zero to `1.5e-4` over 3,000 steps and then
cosine-decays toward 10% of the base rate rather than zero.

Forward and loss computation use CUDA FP16 autocast. `torch.amp.GradScaler`
scales gradients. Each microbatch loss is divided by eight before backward so
the accumulated gradient represents the mean over the effective batch.

An optimizer step is committed only when all eight accumulated microbatches
have finite losses. If any microbatch is non-finite:

- the entire accumulated gradient is discarded;
- the optimizer does not step;
- the scaler is reduced;
- the same global-step number is retried.

After a valid accumulation:

1. gradients are unscaled;
2. global gradient norm is clipped to 1.0;
3. AdamW updates parameters;
4. the scaler updates;
5. CUDA is synchronized for accurate cumulative GPU-time measurement.

## 17. Validation

Pretraining validation is enabled:

```yaml
validation:
  enabled: true
  holdout_samples: 10000
  eval_steps: 500
  eval_batches: 8
```

Every 500 optimizer steps, the trainer evaluates eight held-out batches under
inference mode and the same autocast dtype. It reports:

```text
mean held-out cross-entropy
perplexity = exp(mean loss)
number of supervised validation tokens
```

The model returns to training mode immediately after validation. Validation
records are written separately from training records in the JSONL log through
their `"type": "validation"` field.

## 18. Checkpoint and resume contract

The active pretraining checkpoint is:

```text
checkpoints/tlgm_1b_1024ctx_pretrain20b.pth
```

Only one current checkpoint is retained. Every 500 steps, the new checkpoint
is written atomically and replaces the prior current checkpoint. A JSON
sidecar at the same base path provides human-readable status.

The checkpoint contains:

```text
model
optimizer
scaler
sampler_state
rng_state
global_step
config
metadata
```

Model tensors are converted to FP16 CPU tensors for checkpoint storage.
Optimizer state is retained so AdamW momentum and variance estimates continue
correctly.

Random state includes:

```text
Python random state
NumPy random state
PyTorch CPU RNG state
all CUDA RNG states
```

On startup:

1. if the configured checkpoint exists, it is loaded automatically;
2. model, optimizer, scaler, sampler, and RNG states are restored;
3. global step, cumulative time, and token counters continue;
4. if no checkpoint exists and no SFT initialization checkpoint is specified,
   training begins from random initialization.

Checkpoint metadata includes:

```text
step and max_steps
last loss and learning rate
tokens per optimizer step
total seen tokens
total supervised tokens
cumulative measured GPU seconds and hours
architecture identifier
```

Atomic replacement prevents an interrupted save from leaving a partially
written current checkpoint.

## 19. SFT representation

SFT input is JSONL. Each row has a `conversations` array:

```json
{
  "conversations": [
    {"role": "user", "content": "What is 2+3?"},
    {"role": "assistant", "content": "2+3 is 5."}
  ]
}
```

Rows without a non-empty assistant response are excluded.

The rendered token sequence is conceptually:

```text
<bos>
User: What is 2+3?
Assistant: 2+3 is 5.
<eos>
```

Label policy:

| Region | Label |
|---|---|
| `<bos>` | `-100` |
| User role and user content | `-100` |
| `Assistant: ` prefix | `-100` |
| Assistant answer | corresponding token ID |
| `<eos>` | token ID 2 |
| Padding | `-100` |

If a rendered conversation exceeds 1,024 tokens, the dataset retains the most
recent 1,024 tokens so the assistant answer is preserved. The first retained
position is replaced with `<bos>` and ignored in the loss. Shorter samples are
right-padded with `<pad>`.

## 20. SFT stages

Post-training is a three-stage checkpoint chain:

| Stage | Initialization | Output | Steps | LR |
|---|---|---|---:|---:|
| Chat mix | Pretrain checkpoint | `tlgm_1b_1024ctx_sft_chatmix.pth` | 12,000 | 3e-6 |
| Repair | Chat-mix checkpoint | `tlgm_1b_1024ctx_sft_repair.pth` | 3,000 | 1.5e-6 |
| Polish | Repair checkpoint | `tlgm_1b_1024ctx_sft_final.pth` | 800 | 5e-7 |

All three stages use:

```text
context length:                 1024
microbatch size:                4
gradient accumulation:          16
effective sequences per step:   64
dtype:                          float16
gradient clipping:              1.0
weight decay:                   0.01
```

Each stage uses a different output checkpoint, so the pretrained model and
earlier SFT stages remain available. When an SFT output checkpoint does not
exist, the trainer loads the stage's `init_checkpoint` model weights and
starts a new optimizer schedule for that stage. If its own output exists, it
resumes the complete stage state.

## 21. Autoregressive inference

At each generation iteration:

1. retain at most the newest 1,024 input IDs;
2. run the complete model over that context;
3. take logits from the final position;
4. apply temperature scaling;
5. apply top-k filtering;
6. apply nucleus/top-p filtering;
7. sample one token, or use argmax when temperature is zero;
8. append the token;
9. stop on `<eos>` or after `max_new_tokens`.

Top-k retains at most the `k` highest-logit tokens. Top-p sorts probabilities
and keeps the smallest prefix whose cumulative mass reaches the configured
threshold, always retaining at least one token.

The chat prompt format is:

```text
User: <message>
Assistant:
```

Optional history repeats earlier `User:` and `Assistant:` turns. The CLI
removes decoded text after `<eos>`, `<pad>`, `<bos>`, or a generated `User:`
marker.

After all SFT stages complete, interactive chat runs with:

```powershell
python chat_cli.py
```

The CLI defaults to:

```text
configs/sft_tlgm_1b_polish.yaml
checkpoints/tlgm_1b_1024ctx_sft_final.pth
```

## 22. Computational characteristics

### 22.1 Sequence mixing

A causal sequence linear computes a dense matrix operation over positions for
each feature channel:

```text
O(B * D * N^2)
```

There are two such operations per block and 27 blocks.

### 22.2 Feature mixing

The two feature projections per block cost:

```text
O(B * N * D * H)
```

With `D=2048` and `H=8192`, feature MLP computation and parameters dominate
the model.

### 22.3 Prefill

Prefill processes the entire prompt in parallel. It does not allocate an
attention score tensor and does not create a KV cache. However, each global
block still performs dense sequence-position mixing.

### 22.4 Decoding

The current generator recomputes the complete retained context for every new
token. There is no incremental state cache. If generation length grows from
one to `M`, repeated full-context work makes decoding substantially slower
than a Transformer implementation with an optimized KV cache.

### 22.5 Context length

Learned position matrices and embeddings are fixed at 1,024. Supporting a
larger context is not a runtime-only change: it requires larger position
embeddings and larger sequence matrices, followed by training those new
parameters.

## 23. Architectural constraints

These are properties of the design, not training faults:

- The fixed absolute-position matrices do not naturally extrapolate beyond
  1,024 tokens.
- Sequence routing is content-independent; token content changes feature
  values but does not dynamically choose the position-mixing coefficients.
- There is no KV cache or equivalent incremental decoding state.
- Dense `N x N` position matrices make sequence mixing quadratic in context
  length.
- An 8,192-token vocabulary reduces embedding and output cost but may use more
  tokens per English passage than a larger-vocabulary model.
- A 20B-token budget is meaningful for a 1B experimental model but remains
  much smaller than the budgets used by major commercial language models.
- Model quality ultimately depends on corpus quality, optimization, and SFT
  coverage in addition to architecture and parameter count.

## 24. Rebuild procedure

An independent implementation can reproduce the model by following these
steps exactly:

1. Train or load the exact 8,192-entry byte-level BPE tokenizer with special
   IDs 0 through 3 as specified.
2. Create token and absolute-position embeddings with widths of 2,048.
3. Add them and apply zero-rate dropout.
4. Apply the shared `2048 -> 8192 -> 2048` GELU encoder and LayerNorm.
5. Create 27 independent global blocks.
6. In each block, pre-normalize and apply two learned 1,024 x 1,024
   lower-triangular sequence operators with GELU between them.
7. Add that result residually.
8. Pre-normalize and apply a `2048 -> 8192 -> 2048` GELU feature MLP.
9. Add that result residually.
10. Apply a final 2,048-wide LayerNorm.
11. Project to 8,192 vocabulary logits using the transposed token embedding.
12. Initialize general weights with standard deviation 0.02 and apply the
    specified `1/sqrt(2L)` residual-output scaling.
13. Train with one model-level causal shift:
    `logits[:, :-1]` against `labels[:, 1:]`.
14. Use assistant-only labels for SFT and `-100` elsewhere.
15. Match the optimizer, LR schedule, accumulation, validation, and checkpoint
    contract in this document.

The resulting unique trainable parameter count must be exactly:

```text
1,015,592,960
```

Any different count means at least one dimension, bias, normalization tensor,
weight-sharing rule, or block count does not match this implementation.

## 25. Verification checklist

The source tree includes automated checks covering:

- exact parameter estimator agreement;
- causal prefix invariance;
- one-token next-token label alignment;
- deterministic resumable sampling;
- SFT assistant-label preservation;
- checkpoint/RNG behavior;
- finite initialized forward loss;
- 20B corpus metadata and binary-size consistency.

Recommended verification commands:

```powershell
python -m unittest discover -s tests -v
python scripts\validate_pretrain_data.py
python scripts\check_full_model_init.py
```

Expected full-model initialization properties:

```text
parameters:       1,015,592,960
embedding std:    approximately 0.02
forward loss:     finite
device:           CUDA when available
```

This document describes the current implementation and is the reference for
architecture review, checkpoint compatibility, reproduction, and future
modification.
