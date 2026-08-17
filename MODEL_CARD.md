# TLGM Model Family Card

## Summary

TLGM is a family of English autoregressive causal language models based on
learned lower-triangular sequence mixing rather than self-attention. The
released tracks contain approximately 118M, 121M, and 1.016B parameters.

The project is intended for architecture research, training education,
controlled benchmarking, and experimentation with attention-free language
models. It is not intended as a drop-in replacement for production chat
assistants.

## Provenance

- Initialization: random.
- Pretrained model weights used: none.
- Tokenizer: 8,192-entry byte-level BPE trained from scratch.
- Main language: English.
- Training objective: next-token causal cross-entropy.
- Chat objective: assistant-answer-only masked SFT.
- Final 1B checkpoint: `tlgm_1b_1024ctx_sft_reasoning_3day.pth`.

## Architecture

The 1B model has 27 residual global mixing blocks, width 2,048, feature width
8,192, learned absolute positions, and tied input/output embeddings. Each
block applies two lower-triangular causal sequence matrices shared over
channels, followed by a position-wise MLP. It contains no QKV projections,
softmax attention, Transformer attention head, or KV cache.

## Training

The 1B base model completed approximately one pass over a 20B-token corpus.
Post-training chained chat-mix, repair, polish, logic, and reasoning stages.
The final reasoning stage processed 7.864B padded context positions, of which
2.677B were supervised assistant target tokens, over 88.68 measured GPU hours.

See [DATASETS.md](DATASETS.md) for source lineage and
[reports/1b/TLGM_1B_1024CTX_FULL_PAPER.txt](reports/1b/TLGM_1B_1024CTX_FULL_PAPER.txt)
for the complete build specification.

## Evaluation

The final 1B model achieved 42.33% weighted raw accuracy and 43.98% weighted
length-normalized accuracy over 20,110 questions spanning ARC-Easy,
HellaSwag, PIQA, BoolQ, WinoGrande, OpenBookQA, and TruthfulQA MC1.

On the common-text perplexity protocol it achieved token perplexity 18.406
and 1.1845 bits per UTF-8 byte. This trails TinyLlama-1.1B, OLMo-1B,
Qwen2.5-1.5B, and SmolLM2-1.7B. The reasoning continuation improved over the
original TLGM SFT checkpoint, which scored token perplexity 22.266 and 1.2619
bits per byte.

These results do not establish general intelligence. Multiple-choice
benchmarks do not measure all chat quality, safety, factuality, or coding.

## Intended Uses

- Study of fixed causal global mixing.
- Reproduction of from-scratch language-model training.
- Comparison against attention and recurrent architectures.
- Low-stakes text-generation demonstrations.
- Fine-tuning research where outputs are independently verified.

## Out-of-Scope Uses

- High-stakes decisions or advice.
- Autonomous agents with real-world authority.
- Unsandboxed code execution.
- Factual retrieval without external verification.
- Safety-critical, medical, legal, or financial deployment.
- Claims that the model is equivalent to ChatGPT or established 1B models.

## Risks And Limitations

- Hallucination and factual errors are common.
- Complex logic and multi-step arithmetic are unreliable.
- Responses may repeat templates or confuse nearby concepts.
- The model can reproduce undesirable patterns from public web data.
- Safety tuning is limited and has not received production red-teaming.
- Context is limited to 1,024 tokens.
- Fixed position routing cannot dynamically select earlier content.
- Generation recomputes context for every output token.
- English dominates the data; other languages are unsupported.

Users must validate outputs and add application-specific safety controls.
