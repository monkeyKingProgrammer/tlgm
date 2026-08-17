# Historical Pre-Fix TLGM 1B Architecture Audit

> Historical record only. This report describes the implementation before the
> July 2026 corrections. It must not be used as the specification for the
> current model. See `ARCHITECTURE_AUDIT.md` for the authoritative current
> architecture and training specification.

## 1. Scope and audit result

This document specifies the exact model implemented in `tlgm_1b_1024ctx`.
It is based on the source code, not only the intended design. It covers:

- tokenizer and special-token IDs;
- exact model graph and all tensor shapes;
- mathematical definition of every learned operation;
- parameter accounting down to each tensor;
- initialization behavior;
- pretraining and SFT data construction;
- loss, optimizer, scheduler, AMP, accumulation, and checkpoint behavior;
- autoregressive generation and chat formatting;
- computational and memory characteristics;
- deviations and defects found in the code;
- an exact rebuild checklist.

The configured model contains exactly **1,015,592,960 unique trainable
parameters**. It is not a Transformer. It has no attention, Q/K/V
projections, attention softmax, recurrent state, convolution, Mamba block, or
KV cache. Sequence communication is performed by learned, fixed,
lower-triangular position-mixing matrices.

The model graph and parameter report are internally consistent, and a direct
causality test gave a maximum changed-prefix difference of `0.0`. The initial
audit found one critical training defect:

> The pretraining dataset shifts labels once, and the model loss shifts them
> again. The stopped original pretraining run therefore predicted token `t+2`, not the
> standard next token `t+1`. SFT uses a different, correct one-token objective.

This defect did not change the architecture or parameter count, but it changed
what the original checkpoint learned. It is now fixed in the source. The
original checkpoint is retained only as an invalid historical artifact and
must not be resumed.

## 2. Authoritative source map

The architecture is distributed across these files:

| Responsibility | Source |
|---|---|
| Configuration and parameter formulas | `tlgm/config.py` |
| Top-level causal LM | `tlgm/model.py` |
| Token and position embeddings | `tlgm/embeddings.py` |
| Shared per-token encoder | `tlgm/local_encoder.py` |
| Global mixer blocks | `tlgm/global_mixer.py` |
| Causal position matrix | `tlgm/causal_linear.py` |
| Pretraining and SFT datasets | `tlgm/dataset.py` |
| Training and checkpoints | `tlgm/trainer.py` |
| LR schedule and atomic saves | `tlgm/utils.py` |
| Sampling | `tlgm/generation.py` |
| Tokenizer creation/loading | `tlgm/tokenizer.py` |
| Exact 1B pretraining settings | `configs/pretrain_tlgm_1b_20b.yaml` |
| Three SFT stages | `configs/sft_tlgm_1b_*.yaml` |
| 20B corpus creation | `scripts/prepare_pretrain20b_tokens.py` |
| Interactive prompt format | `chat_cli.py` |

## 3. Exact configuration

The 1B model uses:

```yaml
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
```

Symbols used below:

- `B`: batch size;
- `N`: current sequence length, with `1 <= N <= 1024`;
- `V = 8192`: vocabulary size;
- `E = 2048`: embedding width;
- `D = 2048`: model width;
- `H = 8192`: shared-token hidden width;
- `F = 8192`: feature-MLP hidden width;
- `L = 27`: number of global mixer blocks.

Because `E == D`, the configured input projection is `Identity`. Because
`tie_embeddings` is true and `E == D`, the output head uses the token embedding
matrix and has no separate weight.

## 4. End-to-end graph

The exact forward graph is:

```text
input_ids [B,N]
    |
    +--> token embedding lookup [B,N,2048]
    |
    +--> learned absolute position lookup [1,N,2048]
    |
    +--> elementwise addition and embedding dropout
           |
           v
       z [B,N,2048]
           |
           v
       input projection (Identity)
           |
           v
       shared per-token encoder, once
           |
           v
       h0 [B,N,2048]
           |
           v
       27 independent GlobalMixingBlock instances
           |
           v
       h27 [B,N,2048]
           |
           v
       final LayerNorm
           |
           v
       tied vocabulary projection
           |
           v
       logits [B,N,8192]
```

There is one shared token encoder before the mixer stack. "Shared" means that
the same MLP is applied independently at every sequence position. It is not
repeated or weight-shared across the 27 global blocks.

The 27 global blocks are separate modules with separate parameters. Within a
block, each causal position matrix is shared across all 2048 feature channels.
The feature MLP is shared across all sequence positions.

## 5. Tokenizer

The tokenizer is a Hugging Face `ByteLevelBPETokenizer`, trained from scratch.
The saved artifacts are:

```text
tokenizer/vocab.json
tokenizer/merges.txt
```

The verified vocabulary size is `8192`, with:

```text
<pad> = 0
<bos> = 1
<eos> = 2
<unk> = 3
```

Training uses byte-level BPE with:

```text
vocab_size = 8192
min_frequency = 2
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
```

The tokenizer training script extracts up to 200,000 text samples by default,
replaces internal newlines with spaces, and trains BPE on the resulting UTF-8
text file.

Pretraining does not rely on the tokenizer automatically inserting special
tokens. It explicitly constructs:

```text
[bos_id] + tokenizer.encode(document).ids + [eos_id]
```

SFT and inference also insert the numeric BOS/EOS IDs explicitly.

## 6. Embedding layer

For input token ID `x[b,i]` and absolute position `i`, the embedding output is:

```text
z[b,i,:] = TokenEmbedding[x[b,i],:] + PositionEmbedding[i,:]
```

Shapes:

```text
TokenEmbedding:    [8192, 2048]
PositionEmbedding: [1024, 2048]
input_ids:         [B, N]
z:                 [B, N, 2048]
```

Position IDs always begin at zero for each forward call. They are generated as
`0, 1, ..., N-1`, copied across the batch. The position embeddings are learned
absolute embeddings, not sinusoidal, rotary, relative, or ALiBi embeddings.

Configured embedding dropout is zero.

## 7. Shared token encoder

The shared encoder processes every token independently:

```text
u[b,i] = W1 z[b,i] + b1
v[b,i] = GELU(u[b,i])
w[b,i] = Dropout(v[b,i])
r[b,i] = W2 w[b,i] + b2
h0[b,i] = LayerNorm(r[b,i])
```

Dimensions:

```text
W1: [8192, 2048]
b1: [8192]
W2: [2048, 8192]
b2: [2048]
LayerNorm gamma: [2048]
LayerNorm beta:  [2048]
```

There is no residual connection around this encoder. It cannot move
information between positions because both linear layers operate only on the
last dimension. The same weights are applied at every position.

## 8. Causal sequence linear operator

`CausalSequenceLinear` is the defining sequence operation.

Each instance owns:

```text
raw weight W: [1024, 1024]
bias b:       [1024]
mask M:       [1024, 1024], lower triangular, not checkpointed
```

For a call with current sequence length `N`, the active matrix is:

```text
C = W[0:N, 0:N] * M[0:N, 0:N]
```

Therefore:

```text
C[i,j] = W[i,j] when j <= i
C[i,j] = 0      when j > i
```

For input `X` with shape `[B,N,D]`, output `Y` is:

```text
Y[b,i,d] = sum over j=0..i of X[b,j,d] * C[i,j] + b[i]
```

The implementation is:

```python
y = torch.einsum("bjd,nj->bnd", x, weight)
y = y + bias.view(1, N, 1)
```

Important properties:

1. It is causal because row `i` can only read positions `0..i`.
2. It mixes positions but not feature channels.
3. The same position coefficients are used for every feature channel.
4. The coefficients depend on layer and absolute positions, not token content.
5. It has no attention normalization or content-dependent routing.
6. It supports shorter prefixes by taking the top-left `N x N` submatrix.
7. It rejects any sequence longer than 1024.
8. It performs a dense matrix operation even though the upper triangle is
   masked, so current compute remains quadratic in sequence length.

Two causal matrices composed together remain causal.

## 9. One global mixing block

Each of the 27 blocks has two residual sublayers.

Given block input `X_l` with shape `[B,N,2048]`, the token branch is:

```text
A = LayerNorm_token(X_l)
B = CausalLinear_1(A)
C = GELU(B)
D = CausalLinear_2(C)
X_mid = X_l + D
```

The feature branch is:

```text
E = LayerNorm_feature(X_mid)
F1 = Wf1 E + bf1
F2 = GELU(F1)
F3 = Dropout(F2)
F4 = Wf2 F3 + bf2
F5 = Dropout(F4)
X_(l+1) = X_mid + F5
```

Feature dimensions:

```text
Wf1: [8192, 2048]
bf1: [8192]
Wf2: [2048, 8192]
bf2: [2048]
```

Both norms are pre-norm LayerNorms over the final dimension. Each has learned
gamma and beta vectors of length 2048. Configured dropout is zero, so both
dropout calls are identities in this model.

After the first causal operator in the first block, position `i` has a direct
linear path to every position `0..i`. The feature MLP then nonlinearly combines
the 2048 channels at each position. Repeating this process 27 times provides
deep content processing, but position routing itself remains fixed after
training and independent of input content.

## 10. Final normalization and output head

After block 27:

```text
Q = LayerNorm_final(X_27)
logits[b,i,v] = sum over d of Q[b,i,d] * E_token[v,d]
```

The final LayerNorm has gamma and beta of shape `[2048]`.

The vocabulary projection is an `nn.Linear(2048, 8192, bias=False)`, but its
weight object is replaced by the token embedding weight:

```python
self.lm_head.weight = self.embeddings.token_embedding.weight
```

This is true weight tying. Updating the output projection updates the token
embedding and vice versa. The output head adds zero unique parameters.

## 11. Causality

The model is structurally causal:

- embeddings at position `i` use only token `i` and absolute position `i`;
- the shared encoder is position-local;
- each position matrix is explicitly lower triangular;
- feature MLPs are position-local;
- residuals do not introduce new dependencies;
- final normalization and vocabulary projection are position-local.

An executable test changed all input tokens after position 4 and compared
logits through position 4. The maximum absolute difference was exactly `0.0`.

There is no stochastic dropout in the configured model. In evaluation mode,
identical prefixes therefore produce identical prefix logits.

## 12. Exact parameter accounting

### Embeddings

```text
token embedding:    V * E = 8192 * 2048 = 16,777,216
position embedding: N * E = 1024 * 2048 =  2,097,152
```

### Shared token encoder

```text
first linear weight:  2048 * 8192 = 16,777,216
first linear bias:                    8,192
second linear weight: 8192 * 2048 = 16,777,216
second linear bias:                   2,048
LayerNorm gamma and beta:             4,096
shared encoder total:            33,568,768
```

### One global block

Two causal position operators:

```text
2 * (1024 * 1024 + 1024) = 2,099,200
```

Feature MLP:

```text
2048 * 8192 + 8192 + 8192 * 2048 + 2048 = 33,564,672
```

Two LayerNorms:

```text
2 * (2048 gamma + 2048 beta) = 8,192
```

One block:

```text
2,099,200 + 33,564,672 + 8,192 = 35,672,064
```

All 27 blocks:

```text
27 * 35,672,064 = 963,145,728
```

### Final total

| Component | Parameters |
|---|---:|
| Token embedding | 16,777,216 |
| Position embedding | 2,097,152 |
| Input projection | 0 |
| Shared token encoder | 33,568,768 |
| 27 global blocks | 963,145,728 |
| Final LayerNorm | 4,096 |
| Unique output-head parameters | 0 |
| **Total** | **1,015,592,960** |

Raw parameter storage:

```text
FP32: 3,874.18 MiB
FP16: 1,937.09 MiB
BF16: 1,937.09 MiB
```

These sizes exclude gradients, optimizer states, activations, allocator
overhead, the causal masks, and checkpoint serialization overhead.

## 13. Initialization

The corrected implementation applies explicit initialization:

- token and position embeddings use `Normal(0, 0.02)`;
- ordinary linear weights use `Normal(0, 0.02)`;
- ordinary linear biases start at zero;
- LayerNorm gamma starts at one and beta at zero;
- the first causal matrix in every block uses Xavier uniform with gain 1;
- the second causal matrix uses Xavier uniform with gain
  `1 / sqrt(2 * 27)`;
- the second feature-MLP linear in each residual block uses
  `Normal(0, 0.02 / sqrt(2 * 27))`;
- causal matrix biases start at zero;
- the tied output head shares the corrected token embedding initialization.

The `initializer_range` and residual scaling switch are explicit configuration
fields. This replaces the original unit-standard-deviation embedding defaults,
which produced initial loss near 173. Regression tests now require finite
forward/backward behavior and a small-model initial loss below 10.

## 14. Pretraining corpus

The completed binary stream contains exactly 20,000,000,000 `uint16` token
IDs, occupying 40,000,000,000 decimal bytes.

The corpus quotas are:

| Share | Tokens | Dataset |
|---:|---:|---|
| 40% | 8,000,000,000 | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` |
| 25% | 5,000,000,000 | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` |
| 20% | 4,000,000,000 | `wikimedia/wikipedia`, `20231101.en` |
| 15% | 3,000,000,000 | `HuggingFaceFW/fineweb-edu`, `sample-10BT` |

Rows shorter than 80 characters are skipped. Every accepted document becomes:

```text
<bos> document tokens <eos>
```

Documents are concatenated without padding. If a source quota ends inside a
document, that document is truncated exactly at the quota. Sources are written
in the table's order, but training windows are shuffled by the DataLoader.

The token stream is memory-mapped. Resume progress is written every 10 million
tokens. The completed metadata records token count, type, output path, and
source quotas.

## 15. Pretraining windows and corrected objective

`TokenBinDataset` divides the stream into non-overlapping 1024-token windows:

```text
chunk = stream[start : start + 1024]
input_ids = chunk
labels    = input_ids.clone()
start     = index * 1024
```

The dataset length is:

```text
floor(number_of_tokens / 1024)
= 19,531,250 windows
```

The top-level model then computes:

```python
cross_entropy(
    logits[:, :-1].reshape(-1, V),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

That produces standard next-token alignment:

```text
logit at input position 0 is trained against stream token 1
logit at input position 1 is trained against stream token 2
...
```

```text
P(x_(t+1) | x_0 ... x_t)
```

The original implementation read 1025 tokens, returned pre-shifted labels, and
then shifted again in the model, producing `t+2`. That behavior is removed.

## 16. SFT representation and objective

SFT reads JSONL objects of this form:

```json
{
  "conversations": [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hello! How can I help you?"}
  ]
}
```

The rendered token sequence is conceptually:

```text
<bos>User: hello
Assistant: Hello! How can I help you?
<eos>
```

Exact label rules:

- BOS: `-100`;
- all user-role tokens: `-100`;
- `Assistant: ` prefix: `-100`;
- assistant content plus newline: token IDs themselves;
- EOS: EOS token ID;
- right padding: input ID 0 and label `-100`.

The model-side one-token shift makes this SFT objective correct: each token
before an assistant answer predicts the next assistant token. User and prefix
positions do not directly contribute to cross entropy.

Samples are hard-truncated on the right at 1024 tokens and padded to exactly
1024. There is no attention mask. Because sequence mixing is causal, right-side
pad tokens cannot affect earlier supervised positions. However, a long prompt
can truncate the assistant answer and leave few or no supervised targets.

The three configured SFT stages are:

| Stage | Steps | Sequences/step | Seen tokens |
|---|---:|---:|---:|
| Chat mix | 12,000 | 64 | 786,432,000 |
| Repair | 3,000 | 64 | 196,608,000 |
| Polish | 800 | 64 | 52,428,800 |
| **Total** | **15,800** | | **1,035,468,800** |

"Seen tokens" includes pads and ignored prompt tokens. It is not the number of
assistant tokens used in the loss.

## 17. Training algorithm

### Pretraining settings

```text
microbatch size: 8 sequences
gradient accumulation: 8 microbatches
effective batch: 64 sequences
context: 1024
nominal tokens/optimizer step: 65,536
steps: 305,176
nominal seen tokens: 20,000,014,336
precision: FP16 autocast with GradScaler
optimizer: AdamW
base learning rate: 1.5e-4
weight decay: 0.1
warmup: 3,000 optimizer steps
gradient clipping: global norm 1.0
checkpoint interval: 500 steps
```

AdamW uses PyTorch defaults for unspecified values:

```text
betas = (0.9, 0.999)
epsilon = 1e-8
```

Weight decay is applied through one parameter group to all trainable
parameters, including embeddings, biases, and LayerNorm parameters.

One optimizer step is:

```text
zero gradients
repeat accumulation_count times:
    fetch shuffled microbatch
    move input and labels to GPU
    run FP16 autocast forward
    divide loss by accumulation_count
    scaled backward
set LR from current optimizer-step number
unscale gradients
clip global gradient norm to 1.0
GradScaler optimizer step
GradScaler update
synchronize CUDA
append JSONL loss record
optionally replace checkpoint
```

The LR schedule is:

```text
if step <= warmup:
    lr = base_lr * step / warmup_steps
else:
    progress = (step - warmup) / (max_steps - warmup)
    lr = base_lr * [0.1 + 0.45 * (1 + cos(pi * progress))]
```

The final LR is 10% of the base LR, not zero.

A deterministic tail holdout is excluded from the resumable training sampler.
At configured evaluation intervals, the trainer records target-weighted
validation loss, perplexity, and supervised validation-token count.

### Stage transitions

Pretraining starts from random initialization unless its own checkpoint exists.
Each SFT stage:

1. creates a fresh model and optimizer;
2. loads model weights from the prior stage's checkpoint;
3. does not load the prior optimizer or scaler;
4. starts its own step count and scheduler at zero;
5. writes a separate stage checkpoint.

## 18. Checkpoint format and resume behavior

Each checkpoint is a PyTorch dictionary:

```text
model: model state dict, converted to FP16 on CPU
optimizer: AdamW state dict
scaler: GradScaler state dict
sampler_state: sampler seed, epoch, and exact sample offset
rng_state: Python, NumPy, CPU PyTorch, and all CUDA RNG states
global_step: optimizer step
config: complete YAML-derived configuration dictionary
metadata:
    step
    max_steps
    last_loss
    last_lr
    tokens_per_step
    total_tokens_trained
    cumulative_gpu_time_seconds
    cumulative_gpu_time_hours
    architecture
```

The checkpoint is written to a temporary sibling and atomically replaces the
single final path. This preserves only one logical checkpoint, although disk
usage temporarily includes both old and new checkpoint files during saving.

On same-stage resume, model, optimizer, scaler, global step, cumulative time,
all RNG states, and sampler position are loaded. The scheduler is reconstructed
from global step. The sampler uses an O(1)-memory deterministic affine
permutation and resumes from the exact epoch and sample offset rather than
replaying the beginning of a shuffled stream.

`cumulative_gpu_time_seconds` is synchronized wall-clock training-session time,
including data waits and checkpoint writing. It is not hardware-reported GPU
active time.

## 19. Generation

Generation is fully autoregressive and has no cache:

```text
for each new token:
    take the last at most 1024 token IDs
    run the complete 1B model on that whole context
    select logits from the final position
    apply temperature
    apply top-k filter
    apply nucleus top-p filter
    sample one token, or argmax when temperature <= 0
    append token
    stop if token == EOS
```

Top-k keeps logits at least as large as the kth-largest logit. Top-p sorts
remaining logits, computes softmax probabilities, and removes tokens after the
smallest prefix whose cumulative probability reaches `top_p`, while always
keeping at least one token.

When generation exceeds 1024 tokens of total context, only the newest 1024 IDs
are sent to the model. Absolute positions are then reassigned from zero within
that sliding window. This differs from continuing global absolute positions.

There is no KV cache or reusable state. Every generated token recomputes all 27
blocks for the full retained context.

Chat renders:

```text
User: first message
Assistant: first reply
User: current message
Assistant:
```

By default `history_turns=0`, so only the current user message is included.
Generation stops on EOS at token level. Post-processing also truncates decoded
text at `<eos>`, `<pad>`, `<bos>`, or the string `User:`.

## 20. Compute and memory characteristics

Ignoring elementwise operations, one forward pass is dominated by:

```text
shared encoder:       approximately 2 * B*N*D*H multiplies
each token branch:    approximately 2 * B*D*N^2 multiplies
each feature branch:  approximately 2 * B*N*D*F multiplies
output head:          approximately B*N*D*V multiplies
```

At `B=1`, `N=1024`, `D=2048`, `H=F=8192`, the approximate multiply-accumulate
counts are:

```text
shared encoder: 34.36 billion
one token branch: 4.29 billion
one feature branch: 34.36 billion
27 global blocks: about 1.044 trillion
output projection: 17.18 billion
total forward: about 1.095 trillion multiply-accumulates
```

If one multiply-accumulate is counted as two FLOPs, this is roughly 2.19
TFLOPs per full-length, single-sequence forward pass before implementation
overheads.

Complexity:

```text
token mixing: O(L * B * D * N^2)
feature mixing: O(L * B * N * D * F)
generation: full forward repeated for every new token
```

The current RTX PRO 6000 run uses approximately 34.5 GiB VRAM with microbatch
8, FP16 autocast, and AdamW, while sustaining approximately 24K tokens/second.
FP16 inference weights alone require about 1.9 GiB, but actual inference also
needs activations, logits, framework allocations, and checkpoint-loading
memory.

## 21. Audit findings

### Fixed critical defect: pretraining predicted two tokens ahead

The original `TokenBinDataset` created already-shifted labels, then
`TLGMForCausalLM` shifted labels again. That pretraining run learned `t+2`;
SFT learned `t+1`.

Impact:

- pretraining and SFT objectives are inconsistent;
- autoregressive next-token quality is likely degraded;
- recorded low pretraining loss does not measure standard next-token loss;
- previous TLGM model weaknesses may partly originate here.

Resolution:

- pretraining labels are now an unshifted copy of input IDs;
- an exact alignment regression test was added;
- the invalid checkpoint is not eligible for resume;
- corrected training must restart from random initialization.

### Fixed high-severity defect: resume repeated earlier data

Checkpoint resume restores optimization state but not RNG or sampler state.
The shuffled stream restarts instead of continuing from the prior sample.

Resolution: all RNG states plus sampler seed, epoch, and offset are checkpointed
and restored. An end-to-end stop/resume regression test checks continuation and
exact token counters.

### Fixed high-severity defect: initialization was poorly scaled

Unit-standard-deviation embeddings and a tied output matrix produced initial
loss around 173 rather than near `log(8192) = 9.01`. The run recovered, but
this spends optimization capacity correcting scale and raises stability risk.

Resolution: explicit `0.02` initialization and `1/sqrt(2L)` residual-output
scaling are implemented and tested.

### Fixed high-severity gap: no validation, perplexity, or regression tests

Held-out validation loss and perplexity are implemented. Automated tests cover
parameter count, tying, causality, finite initialization/backward, next-token
alignment, sampler resume, SFT truncation, and end-to-end checkpoint resume.

Recommended minimum tests:

1. exact next-token label alignment;
2. future-token mutation does not affect prefix logits;
3. parameter total equals 1,015,592,960;
4. tied weights share the same storage;
5. forward and backward are finite;
6. checkpoint-resume produces the same next update;
7. overfit a tiny deterministic corpus;
8. validation next-token loss and perplexity.

### Medium: fixed position mixing limits extrapolation and caching

Every block learns two absolute `1024 x 1024` matrices. The model cannot extend
past 1024 without adding/retraining parameters. It cannot use a standard KV
cache, and sliding-window generation resets position semantics.

### Fixed medium-severity gap: token accounting was nominal

Metadata and JSONL logs now distinguish:

- raw tokens loaded;
- non-pad tokens;
- supervised target tokens;
- optimizer-step-equivalent nominal tokens.

Non-finite microbatches now invalidate the entire accumulated update instead of
silently under-scaling gradients.

### Fixed medium-severity defect: SFT right truncation could remove answers

Long conversations are truncated at 1024 from the right. This may retain the
prompt while discarding assistant targets, potentially creating all-ignored
loss batches.

Samples without assistant responses are rejected. Overlength samples retain
the newest 1024 tokens, preserving the final answer and EOS, and reset the
first retained position to an ignored BOS token.

### Fixed medium-severity gap: corpus completion trusted metadata existence

The pipeline skips data preparation when the metadata file exists, without
validating that it says 20B tokens or that the binary file has the matching
size. Progress-state writes are not atomic.

Progress and completion metadata are now atomic. A dedicated validator checks
target count, source-token sum, dtype-sized binary length, and required files
before training. Full cryptographic checksums are still not generated.

### Partly fixed low-severity API and checkpoint-loading concerns

The trainer now uses `torch.amp`. Checkpoints still require pickle-capable
loading because they contain optimizer, RNG, and NumPy state. Only locally
generated trusted checkpoints may be loaded.

## 22. Exact rebuild recipe

To reproduce the architecture exactly:

1. Use Python, PyTorch, NumPy, PyYAML, `tokenizers`, `datasets`, TensorBoard,
   and tqdm.
2. Train or copy the exact 8192-entry byte-level BPE tokenizer with special IDs
   0 through 3 as listed above.
3. Instantiate `TLGMConfig` with the values in Section 3.
4. Build token and position embeddings and add them.
5. Apply the single shared `2048 -> 8192 -> 2048` GELU MLP followed by
   LayerNorm.
6. Build 27 independent blocks. Each block must contain:
   - pre-token LayerNorm;
   - two independent learned `1024 x 1024` matrices, each masked lower
     triangular on every forward;
   - GELU between those matrices;
   - token residual;
   - pre-feature LayerNorm;
   - `2048 -> 8192 -> 2048` GELU MLP;
   - feature residual.
7. Apply final LayerNorm.
8. Project to 8192 logits with no bias and tie that weight to the token
   embedding.
9. Use standard one-token causal cross entropy. The archived invalid checkpoint
   used pre-shifted labels plus the model-side shift and must not be reproduced.
10. Use the pretraining optimizer and scheduler in Section 17.
11. Save model, optimizer, scaler, global step, configuration, and metadata.
12. For SFT, render exact `User:` and `Assistant:` prefixes and supervise only
    assistant content plus EOS.
13. For generation, rerun the full model on the newest at most 1024 tokens for
    every sampled token.

Equivalent constructor pseudocode:

```python
token_embedding = Embedding(8192, 2048)
position_embedding = Embedding(1024, 2048)

shared_encoder = Sequential(
    Linear(2048, 8192),
    GELU(),
    Dropout(0.0),
    Linear(8192, 2048),
    LayerNorm(2048),
)

blocks = ModuleList([
    GlobalMixingBlock(
        context_length=1024,
        model_dim=2048,
        feature_hidden_dim=8192,
        dropout=0.0,
    )
    for _ in range(27)
])

final_norm = LayerNorm(2048)
lm_head = Linear(2048, 8192, bias=False)
lm_head.weight = token_embedding.weight
```

The implementation is equivalent only if every block has distinct parameters,
the position matrices are lower triangular at runtime, and the output weight
is the same parameter object as the input embedding.

## 23. Rebuild verification checklist

A correct equivalent implementation should satisfy all of these:

```text
[ ] tokenizer vocabulary size is 8192
[ ] special IDs are pad=0, bos=1, eos=2, unk=3
[ ] maximum context is 1024
[ ] model width and embedding width are 2048
[ ] shared hidden width and block feature width are 8192
[ ] there are exactly 27 independent global blocks
[ ] each block has exactly two causal position matrices
[ ] causal matrices are 1024 x 1024 with a 1024-element bias
[ ] every causal matrix is shared across feature channels
[ ] every feature MLP is shared across positions
[ ] there is no attention, QKV, recurrence, convolution, or KV cache
[ ] output and token embedding weights are tied
[ ] unique trainable parameter count is 1,015,592,960
[ ] changing future tokens leaves all earlier logits unchanged
[ ] intended training predicts t+1, not t+2
[ ] checkpoint resume restores both optimizer and data-stream position
[ ] initial loss/logit scale is explicitly tested
[ ] validation next-token loss and perplexity are recorded
```

This checklist, the exact dimensions, equations, initialization rules, and
training semantics above are sufficient to independently reconstruct the same
TLGM 1B architecture and to distinguish the archived invalid training behavior
from the corrected causal language model.
