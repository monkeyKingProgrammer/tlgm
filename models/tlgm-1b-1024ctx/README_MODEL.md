# TLGM 1B 1024-Context Plan

This folder contains the active TLGM 1B experiment and its complete training
pipeline.

## Full Technical Paper

See [TLGM_1B_1024CTX_FULL_PAPER.txt](TLGM_1B_1024CTX_FULL_PAPER.txt) for the
standalone model paper, complete reproduction specification, model card,
training data, hardware, deployment, and timestamped experiment status.

## Current Technical Audit

See [TLGM_1B_COMPLETE_TECHNICAL_AUDIT_2026-08-12.txt](TLGM_1B_COMPLETE_TECHNICAL_AUDIT_2026-08-12.txt)
for the post-training code audit, live reasoning-run status, exact token and
GPU accounting, data lineage, benchmark analysis, external 1B-class model
comparison, and prioritized remediation plan.

## Practical Benchmarks

See [BENCHMARKS.md](BENCHMARKS.md) for the post-training ARC-Easy, HellaSwag,
PIQA, BoolQ, WinoGrande, OpenBookQA, and TruthfulQA evaluation pipeline.

## Architecture Audit

See [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) for the complete as-built
architecture, tensor equations, parameter derivation, training and inference
semantics, reproducibility checklist, and severity-ranked code audit.

The audit identified a critical double shift in the original pretraining
target. The corrected implementation now uses standard one-token causal
labels, explicit scaled initialization, resumable sampler/RNG state, held-out
validation, perplexity, integrity checks, and regression tests.

## Model

Target architecture:

```text
Architecture: TLGM causal lower-triangular sequence mixer
Parameters: 1,015,592,960, about 1.016B
Context length: 1024
Vocab size: 8192
embed_dim/model_dim: 2048
global blocks: 27
local_hidden_dim: 8192
feature_hidden_dim: 8192
attention/QKV: none
KV cache: none
```

Parameter breakdown:

```text
token_embedding:    16,777,216
position_embedding:  2,097,152
shared_encoder:     33,568,768
global_mixer:      963,145,728
final_norm:              4,096
output_head:                 0, tied embeddings
total:          1,015,592,960
```

## Recommended Training Budget

Recommended first serious run:

```text
Pretraining: 20B tokens
SFT chatmix: 786M seen tokens
Repair SFT: 197M seen tokens
Polish SFT: 52M seen tokens
Total post-train/SFT: about 1.03B seen tokens
```

Why 20B? A 1B model usually needs much more data than the current 120M run. A useful rule of thumb is 10B-20B minimum for a 1B learning model, with 20B being the more serious target.

## Dataset Mix

Pretraining mix in `scripts/prepare_pretrain20b_tokens.py`:

```text
40% HuggingFaceTB/smollm-corpus fineweb-edu-dedup: 8B tokens
25% HuggingFaceTB/smollm-corpus cosmopedia-v2:     5B tokens
20% wikimedia/wikipedia 20231101.en:               4B tokens
15% HuggingFaceFW/fineweb-edu sample-10BT:         3B tokens
```

Expected token bin size:

```text
20B uint16 tokens = about 40GB
```

## Runtime Estimate On RTX PRO 6000

Current measured TLGM 120M 1024ctx knowledge pretrain speed:

```text
about 2.0 steps/sec
65,536 tokens/step
about 131k tokens/sec
```

The 1B model is around 8.4x more parameters than 120M. Real speed will not scale perfectly, but a practical estimate is:

```text
Estimated 1B throughput: 15k-30k tokens/sec
Pretraining 20B tokens: 7.7-15.4 days
Likely middle estimate: 10-12 days
Post-train/SFT 1.03B tokens: 10-24 hours
```

If batch size `8` is too large, reduce to:

```yaml
batch_size: 4
gradient_accumulation_steps: 16
```

That keeps the same 65,536 tokens/step but uses less VRAM.

## Files

Configs:

```text
configs/pretrain_tlgm_1b_20b.yaml
configs/sft_tlgm_1b_chatmix.yaml
configs/sft_tlgm_1b_repair.yaml
configs/sft_tlgm_1b_polish.yaml
```

Data prep:

```text
scripts/prepare_pretrain20b_tokens.py
```

Linux one-shot runner:

```text
run_tlgm_1b_full_pipeline.sh
```

Windows remote launcher:

```text
C:\Users\ADMIN\minimind\run_rtx6000_tlgm_1b_full_pipeline.bat
C:\Users\ADMIN\minimind\monitor_rtx6000_tlgm_1b.bat
```

## Automatic Fair Perplexity Comparison

The active `tlgm1b-reasoning.service` is chained to
`tlgm1b-fair-ppl.service` with systemd `OnSuccess`. The comparison starts
only after checkpoint metadata confirms exactly step `120000`; interrupted or
manually stopped training cannot pass this gate.

The comparison evaluates these checkpoints and public base models:

```text
TLGM-1B original SFT
TLGM-1B best reasoning SFT
TLGM-1B final reasoning SFT
TinyLlama-1.1B-3T
OLMo-1B
SmolLM2-1.7B
Qwen2.5-1.5B
```

Every model is evaluated in BF16 on the complete WikiText-2-raw and
WikiText-103-raw test splits with a common 1024-token context. The report
includes token perplexity and bits per UTF-8 byte (BPB). BPB is the primary
cross-tokenizer comparison because ordinary perplexity changes with tokenizer
vocabulary and token boundaries.

Remote status and logs:

```bash
systemctl status tlgm1b-fair-ppl.service
journalctl -u tlgm1b-fair-ppl.service -f
```

Final remote artifacts:

```text
outputs/fair_perplexity_1b/FAIR_PERPLEXITY_REPORT.md
outputs/fair_perplexity_1b/results.json
outputs/fair_perplexity_1b/fair_comparison.png
```

From Windows, download and open the finished comparison with:

```bat
C:\Users\ADMIN\minimind\tlgm_1b_1024ctx\get_fair_comparison.bat
```

The evaluator is resumable after network failures or reboots. To execute it
manually after completed training:

```bash
cd /home/user/minimind/tlgm_1b_1024ctx
source /home/user/venvs/tlgm/bin/activate
python3 scripts/require_completed_reasoning_training.py
python3 scripts/evaluate_fair_perplexity.py
```
