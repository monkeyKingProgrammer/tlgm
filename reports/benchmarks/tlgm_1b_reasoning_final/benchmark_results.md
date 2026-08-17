# TLGM 1B Practical Benchmark Results

- Checkpoint: `/home/user/minimind/tlgm_1b_1024ctx/checkpoints/tlgm_1b_1024ctx_sft_reasoning_3day.pth`
- Evaluated: `2026-08-17 12:54:13 +0800`
- Device: `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Precision: `bfloat16`
- Evaluation batch size: `8`

This is a custom zero-shot conditional log-likelihood protocol. It is useful
for comparisons made with this exact pipeline, but it is not automatically
identical to lm-evaluation-harness or official leaderboard prompting.

| Task | Questions | Accuracy | Accuracy norm | Random | Seconds |
|---|---:|---:|---:|---:|---:|
| arc_easy | 2,376 | 59.26% | 48.95% | 25.02% | 12.7 |
| hellaswag | 10,042 | 32.59% | 37.35% | 25.00% | 89.2 |
| piqa | 1,838 | 66.49% | 66.38% | 50.00% | 6.1 |
| boolq | 3,270 | 53.24% | 53.24% | 50.00% | 26.7 |
| winogrande | 1,267 | 51.07% | 53.12% | 50.00% | 4.2 |
| openbookqa | 500 | 17.60% | 31.40% | 25.00% | 2.3 |
| truthfulqa_mc1 | 817 | 16.40% | 17.01% | 22.61% | 5.4 |

## Aggregate

- Questions: `20,110`
- Weighted accuracy: `42.33%`
- Weighted normalized accuracy: `43.98%`
- Macro task accuracy: `42.38%`
- Macro normalized task accuracy: `43.92%`

Raw accuracy uses the sum of continuation log probabilities. Normalized
accuracy divides each choice score by its number of continuation tokens.
Per-example scores are stored in the corresponding prediction JSONL files.
