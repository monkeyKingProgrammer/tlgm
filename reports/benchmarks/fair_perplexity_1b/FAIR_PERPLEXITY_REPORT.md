# Fair 1B Language-Model Comparison

Generated: `2026-08-17T12:49:30.811119+08:00`

## Protocol

- Common context length: `1024`
- Precision: `bfloat16`
- Dataset text and UTF-8 byte denominator are identical for every model.
- Every non-special text token is scored once with teacher forcing.
- Each non-overlapping context segment starts with the model's BOS token.
- Token perplexity is tokenizer-dependent. Bits per byte (BPB) is the primary cross-tokenizer metric; lower is better.

## WikiText-2-raw

Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`

UTF-8 bytes: `1,296,370`

Text SHA-256: `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`

| Model | Params | Token PPL | Bits/byte | Byte PPL | Tokens/byte | tok/s | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-1.7B | 1.711B | 9.289 | **0.7565** | 1.689 | 0.235 | 51,575 | 5.07 |
| Qwen2.5-1.5B | 1.544B | 10.555 | **0.7844** | 1.722 | 0.231 | 50,977 | 8.68 |
| TinyLlama-1.1B-3T | 1.100B | 8.683 | **0.8213** | 1.767 | 0.263 | 81,282 | 3.28 |
| OLMo-1B | 1.177B | 13.708 | **0.8412** | 1.792 | 0.223 | 73,147 | 4.12 |
| TLGM-1B-reasoning-best | 1.016B | 18.406 | **1.1845** | 2.273 | 0.282 | 85,035 | 2.46 |
| TLGM-1B-reasoning-final | 1.016B | 18.406 | **1.1845** | 2.273 | 0.282 | 84,600 | 2.46 |
| TLGM-1B-original-SFT | 1.016B | 22.266 | **1.2619** | 2.398 | 0.282 | 84,454 | 2.46 |

## WikiText-103-raw

Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`

UTF-8 bytes: `1,296,370`

Text SHA-256: `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`

| Model | Params | Token PPL | Bits/byte | Byte PPL | Tokens/byte | tok/s | Peak VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolLM2-1.7B | 1.711B | 9.289 | **0.7565** | 1.689 | 0.235 | 51,320 | 5.07 |
| Qwen2.5-1.5B | 1.544B | 10.555 | **0.7844** | 1.722 | 0.231 | 50,734 | 8.68 |
| TinyLlama-1.1B-3T | 1.100B | 8.683 | **0.8213** | 1.767 | 0.263 | 82,574 | 3.28 |
| OLMo-1B | 1.177B | 13.708 | **0.8412** | 1.792 | 0.223 | 73,065 | 4.12 |
| TLGM-1B-reasoning-best | 1.016B | 18.406 | **1.1845** | 2.273 | 0.282 | 84,636 | 2.46 |
| TLGM-1B-reasoning-final | 1.016B | 18.406 | **1.1845** | 2.273 | 0.282 | 84,383 | 2.46 |
| TLGM-1B-original-SFT | 1.016B | 22.266 | **1.2619** | 2.398 | 0.282 | 85,162 | 2.46 |

## Interpretation

Token perplexity should only be compared cautiously because vocabulary and token boundaries differ.
BPB divides total negative log-likelihood by the identical number of source bytes and is the fairest metric in this report.
Neither metric directly measures instruction following, factual accuracy, safety, or reasoning; use the practical benchmark report for those capabilities.
