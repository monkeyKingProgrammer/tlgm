
# TLGM-120M: Token-Local Global-Mixing Language Model
## Complete Codex Project Prompt

You are an expert AI researcher and senior PyTorch engineer.

Your task is to build a **complete autoregressive language model from scratch** based on my novel architecture. Do **not** use any Transformer blocks, self-attention, QKV projections, Mamba, RNNs, LSTMs, or GRUs.

## Objective

Implement a causal language model called **TLGM (Token-Local Global-Mixing)** with approximately **120 million trainable parameters**.

### Core pipeline

```
Input Tokens
      │
      ▼
Byte-level BPE Tokenizer (trained from scratch)
      │
      ▼
Token Embedding
      │
      ▼
Position Embedding
      │
      ▼
Shared Token Encoder
      │
      ▼
Global Mixing Network
      │
      ▼
Vocabulary Projection
      │
      ▼
Next-token prediction
```

---

# 1. Tokenization

Use a **Byte-level BPE tokenizer** trained completely from scratch.

Requirements:

- HuggingFace `tokenizers`
- No pretrained tokenizer
- Vocabulary size: 8192
- Special tokens:
  - `<pad>`
  - `<bos>`
  - `<eos>`
  - `<unk>`
- Save tokenizer locally
- Support encode/decode
- UTF-8 safe

Dataset pipeline:

1. Read raw text.
2. Train tokenizer.
3. Encode text.
4. Concatenate token ids.
5. Split into fixed-length windows.
6. Create shifted next-token labels.

---

# 2. Embeddings

Learn:

- token embedding
- positional embedding

Combine:

```
z_i = token_embedding(x_i) + positional_embedding(i)
```

Tensor:

```
[B, Sequence, Embedding]
```

---

# 3. Shared Token Encoder

Every token is processed **independently** using the **same shared network**.

The encoder does **not** mix information between tokens.

Suggested architecture:

Linear
→ GELU
→ Linear
→ LayerNorm

Mathematically:

```
h_i = SharedTokenEncoder(z_i)
```

Output:

```
[B, Sequence, ModelDim]
```

---

# 4. Global Mixing Network

Do NOT flatten the sequence.

Keep:

```
[B, N, D]
```

Each Global Mixing Block contains:

1. LayerNorm
2. Token-position mixing MLP
3. Residual
4. LayerNorm
5. Feature mixing MLP
6. Residual

No attention.

No QKV.

No softmax attention.

No causal attention.

Information should flow through learned dense mixing.

Implement causal masking structurally so future tokens cannot influence earlier positions.

---

# 5. Output Head

Project every position:

```
[B,N,D]
    ↓
Linear
    ↓
[B,N,Vocab]
```

Train using next-token prediction with cross entropy.

Support:

- Greedy decoding
- Temperature
- Top-k
- Top-p

---

# 6. Model Size

Target parameter count:

**120 million trainable parameters**

Acceptable range:

118M–122M

Print:

- embedding parameters
- shared encoder parameters
- global mixer parameters
- output head parameters
- total parameters
- FP32 size
- FP16 size
- BF16 size

Tie embedding weights with output head whenever possible.

---

# 7. Initial Configuration

```
Vocabulary: 8192

Context Length: 256

Embedding Dimension: auto-tune

Model Dimension: auto-tune

Global Blocks: auto-tune

Local Hidden Dimension: auto-tune

Token Mixing Hidden: auto-tune

Feature Mixing Hidden: auto-tune

Target Parameters:
120 million
```

Automatically adjust hidden dimensions until the model is close to 120M.

---

# 8. Training

Implement:

- PyTorch
- CUDA
- AMP
- AdamW
- cosine LR
- warmup
- gradient accumulation
- gradient clipping
- checkpoint resume
- TensorBoard
- validation
- perplexity
- text generation during training

---

# 9. Dataset

Support TinyStories first.

Dataset class should:

- tokenize
- cache
- create windows
- shifted labels

---

# 10. Generation

Implement:

```
generate(
prompt,
temperature,
top_k,
top_p,
max_new_tokens
)
```

---

# 11. Project Layout

```
tlgm/
    config.py
    embeddings.py
    tokenizer.py
    local_encoder.py
    global_mixer.py
    causal_linear.py
    model.py
    trainer.py
    generation.py
    dataset.py
    utils.py

scripts/
    train_tokenizer.py
    prepare_data.py
    train.py
    generate.py
    compare_models.py

tests/
    test_shapes.py
    test_forward.py
    test_generation.py
    test_causality.py
    test_parameters.py
```

---

# 12. Coding Standards

Use:

- Python typing
- docstrings
- dataclasses
- assertions
- unit tests
- readable code

No placeholder implementations.

---

# 13. Deliverables

Produce:

1. Architecture explanation
2. Tensor shape walkthrough
3. Mathematical formulation
4. Complete implementation
5. Training scripts
6. Generation scripts
7. Unit tests
8. README
9. Example commands

The code should run end-to-end and train a TLGM language model completely from scratch using the architecture above.
