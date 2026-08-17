# TLGM 1B Practical Benchmarks

This suite evaluates the final SFT checkpoint with deterministic zero-shot
multiple-choice subsets. It is designed to finish in a practical amount of
time on the RTX PRO 6000 without converting TLGM to another model format.

## Default Suite

| Task | Capability | Default questions |
|---|---|---:|
| ARC-Easy | Grade-school science | 500 |
| HellaSwag | Commonsense continuation | 500 |
| PIQA | Physical reasoning | 500 |
| BoolQ | Reading comprehension | 500 |
| WinoGrande | Coreference and commonsense | 500 |
| OpenBookQA | Elementary science | 500 |
| TruthfulQA MC1 | Truthfulness | 500 |

The maximum default total is 3,500 questions. Some source splits contain
fewer than 500 rows, in which case all available rows are used.

## Scoring

Every answer choice is appended to the same prompt. The model scores only the
choice continuation tokens:

```text
raw score = sum(log P(choice_token | prompt and earlier choice tokens))
normalized score = raw score / number of choice tokens
```

The report includes both raw and normalized accuracy, random-choice baseline,
95% approximate confidence intervals, runtime, peak CUDA memory, and
per-example scores.

This is a custom protocol. Compare models only when they use this same data,
prompt templates, tokenizer boundary handling, and scoring implementation.
Numbers are not automatically identical to official leaderboards or
`lm-evaluation-harness`.

## Run On The RTX PRO 6000

After the complete pretraining and all three SFT stages finish:

```bat
C:\Users\ADMIN\minimind\run_rtx6000_tlgm1b_benchmarks.bat
```

Monitor:

```bat
C:\Users\ADMIN\minimind\monitor_rtx6000_tlgm1b_benchmarks.bat
```

The launcher refuses to evaluate if the final polish checkpoint metadata has
not reached its configured final step.

## Run Directly On Linux

```bash
cd /home/user/minimind/tlgm_1b_1024ctx
bash run_tlgm_1b_benchmarks.sh
```

Runtime controls:

```bash
SAMPLES_PER_TASK=100 EVAL_BATCH_SIZE=8 bash run_tlgm_1b_benchmarks.sh
SAMPLES_PER_TASK=500 EVAL_BATCH_SIZE=16 bash run_tlgm_1b_benchmarks.sh
```

Use 100 examples per task for a quick comparison. The default 500-example
suite should be substantially more stable. If batch size 16 causes a CUDA
out-of-memory error, return to the safe default of 8.

Set `SAMPLES_PER_TASK=0` to prepare full benchmark splits. Full splits take
longer and should use a separate output directory if results from the
500-example protocol must be retained.

## Outputs

```text
outputs/benchmarks/tlgm_1b_final/
  run_metadata.json
  benchmark_results.json
  benchmark_results.md
  <task>_summary.json
  <task>_predictions.jsonl
```

The evaluator saves each completed task independently. Rerunning with
`--resume` skips tasks that already have both summary and prediction files.

## Compare Another Checkpoint

Run the evaluator explicitly:

```bash
python scripts/evaluate_benchmarks.py \
  --config configs/benchmark_tlgm_1b.yaml \
  --checkpoint checkpoints/another_checkpoint.pth
```

Change `output_dir` in a copied benchmark YAML before comparing checkpoints,
otherwise existing resumed task files can be mistaken for the new model's
results.
