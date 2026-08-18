# TLGM-1B-32K Vocabulary, 1024 Context

This is a fresh non-Transformer TLGM language-model experiment with a
32,000-entry tokenizer and 1,024-token context window. Here, **32K means
vocabulary size, not context length**.

The model uses no attention, Q, K, V, pretrained model, inherited checkpoint,
or external tokenizer. Its required pipeline is:

```text
new 32K tokenizer -> random model initialization -> 50B-token pretraining
-> broad chat SFT -> behavior polish -> reasoning SFT -> evaluation -> chat
```

## Model

| Property | Value |
|---|---:|
| Parameters | 1,064,351,744 |
| Vocabulary | 32,000 byte-level BPE entries |
| Context | 1,024 tokens |
| Embedding/model width | 2,048 |
| Global mixer blocks | 27 |
| Feature MLP width | 8,192 |
| Shared local encoder width | 8,192 |
| Token embedding parameters | 65,536,000 |
| Position embedding parameters | 2,097,152 |
| Weight tying | Input embedding and LM head |
| Training precision | BF16 |

The architecture is the corrected TLGM lower-triangular causal sequence mixer.
It applies a shared token encoder, then 27 residual blocks containing learned
causal token mixing and feature MLP mixing. It contains no QKV projections.

Increasing the vocabulary from 8,192 to 32,000 adds 48,758,784 tied embedding
parameters, increasing the old 1.016B model to 1.064B. Weight tying means this
matrix is also the vocabulary projection and is counted only once.

## Budget

Each optimizer step processes `8 x 8 x 1024 = 65,536` pretraining positions.
The 762,940-step schedule therefore processes 50,000,035,840 positions, one
rounded pass over the 50B stream.

| Stage | Steps | Padded/context positions |
|---|---:|---:|
| Pretraining | 762,940 | 50,000,035,840 |
| Broad chat SFT | 12,000 | 786,432,000 |
| Behavior polish | 1,000 | 65,536,000 |
| Reasoning SFT | 120,000 | 7,864,320,000 |
| **Total** | **895,940** | **58,716,323,840** |

SFT loss is masked to assistant responses. Consequently, 8.716B SFT context
positions do not mean 8.716B supervised answer tokens. Based on the previous
run, the final number of supervised targets is expected to be around 2.8-3.2B.

See [DATA_CARD.md](DATA_CARD.md) for exact data sources and quotas.

## Expected Runtime

The previous 8K model processed 20B pretraining positions in 231.7 measured GPU
hours on the 360W-limited NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
The larger vocabulary is expected to reduce training throughput by 5-10%.

| Work | Expected elapsed GPU time |
|---|---:|
| Tokenizer corpus and 50B tokenization | Roughly 8-24 hours, network dependent |
| 50B pretraining | Roughly 25-28 days |
| All SFT stages | Roughly 4.5-5.5 days |
| Practical and TLGM-only perplexity evaluation | Under one hour |
| **End-to-end** | **About 30-34 uninterrupted days** |

These are estimates, not guarantees. Wi-Fi retries, dataset-server latency,
checkpoint writes, validation, thermal/power limits, and reboots add wall time.

## Expected Performance

The completed 8K/20B TLGM obtained 1.1845 bits per byte on the project's common
WikiText evaluation and 42.33% weighted accuracy on its seven-task custom
zero-shot suite. For this 32K/50B run, a defensible pre-run target is:

- WikiText bits per byte: approximately 0.95-1.10.
- Custom seven-task weighted accuracy: approximately 45-49%.
- Better output fluency, token compression, factual coverage, mathematics, and
  instruction stability than the 8K/20B checkpoint.

These ranges are projections and must not be presented as measured results.
The model is still unlikely to match TinyLlama-1.1B or OLMo-1B overall: each was
trained on roughly 3T tokens, about 60 times this model's pretraining budget.
The old fair test measured TinyLlama at 0.8213 BPB and OLMo at 0.8412 BPB. The
new model may narrow the gap, but a vocabulary change does not remove TLGM's
content-independent token-routing limitation or compensate for trillions of
missing training tokens.

## Setup

Recommended system:

- Ubuntu Linux on the RTX PRO 6000 workstation.
- Python 3.11 or 3.12.
- Current NVIDIA driver and CUDA-enabled PyTorch.
- At least 125GB system RAM.
- At least 250GB free disk for the 100GB token stream, caches, checkpoints, and
  temporary files. More is strongly preferred.

```bash
cd /home/user/minimind/tlgm_1b_32k_vocab_1024ctx
python3 -m venv /home/user/venvs/tlgm32k
source /home/user/venvs/tlgm32k/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 check_gpu.py
python3 scripts/verify_project.py
pytest -q
```

## Run Later

The complete pipeline is deliberately prepared but not started yet. When ready:

```bash
cd /home/user/minimind/tlgm_1b_32k_vocab_1024ctx
chmod +x run_full_pipeline.sh run_pretrain_session.sh run_chat.sh
nohup ./run_full_pipeline.sh > outputs/full_pipeline.log 2>&1 &
```

Monitor it with:

```bash
tail -f outputs/full_pipeline.log
```

The pipeline is restartable. Run the same command after a reboot; completed
artifacts are validated and skipped, while each trainer automatically resumes
its own latest checkpoint.

For a bounded pretraining session after tokenizer/data preparation, use:

```bash
./run_pretrain_session.sh 12
```

This trains for at most approximately 12 hours, saves a checkpoint, and exits.

## Manual Stages

```bash
python3 scripts/train_tokenizer.py --vocab_size 32000
python3 scripts/prepare_pretrain50b_tokens.py
python3 scripts/validate_pretrain_data.py
python3 scripts/train.py --config configs/pretrain_tlgm_1b_32k_50b.yaml
python3 scripts/prepare_sft_data.py
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_chat.yaml
python3 scripts/prepare_polish_sft.py --output data/processed/sft_polish_32k.jsonl --repeats 750
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_polish.yaml
python3 scripts/prepare_reasoning_sft.py
python3 scripts/train.py --config configs/sft_tlgm_1b_32k_reasoning.yaml
```

## Chat

After SFT finishes:

```bash
./run_chat.sh
```

## Safety And Limitations

This is an experimental small language model, not ChatGPT. It can hallucinate,
repeat, give unsafe advice, and produce incorrect facts, mathematics, or code.
Its training data is mostly English. Do not rely on it for medical, legal,
financial, security, or other consequential decisions.
