# TLGM-1B-32K Data Card

## Pretraining Mix

The pretraining builder writes exactly 50 billion project-tokenizer tokens.
Every document is delimited with `<bos>` and `<eos>`. Dataset revisions are
resolved before the run, pinned into resumable state, and copied into the final
manifest.

| Source | Share | Token quota | Purpose |
|---|---:|---:|---|
| `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 35% | 17.5B | Deduplicated educational web text |
| `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 25% | 12.5B | Synthetic textbooks, explanations, stories |
| `HuggingFaceFW/fineweb`, `sample-10BT` | 17% | 8.5B | Broader web-domain coverage |
| `open-web-math/open-web-math` | 15% | 7.5B | Mathematics, physics, statistics, technical text |
| `wikimedia/wikipedia`, `20231101.en` | 8% | 4.0B | Factual and encyclopedic English text |

The 100GB `uint16` token binary is generated locally and is not distributed.
Users must review every upstream dataset license and terms before training or
redistributing derivatives. The repository license does not replace upstream
dataset licenses.

## Tokenizer Corpus

The byte-level BPE tokenizer is trained from random corpus statistics, not from
another model. It uses a 4GB stratified sample with the same source weights as
pretraining. It contains exactly 32,000 entries, including:

- `<pad>` = 0
- `<bos>` = 1
- `<eos>` = 2
- `<unk>` = 3

Byte-level BPE can represent arbitrary input bytes. The tokenizer manifest
stores SHA-256 hashes for its corpus shards, vocabulary, and merge table.

## Post-Training Data

Broad chat SFT uses training splits from:

- `HuggingFaceH4/ultrachat_200k`: up to 200,000 conversations.
- `databricks/databricks-dolly-15k`: up to 15,000 instruction examples.
- A small deterministic project-written polish set for greetings, basic facts,
  arithmetic, short descriptions, stories, and uncertainty behavior.

Reasoning SFT uses:

- `nvidia/OpenMathInstruct-2`: up to 1,000,000 examples.
- `open-r1/OpenR1-Math-220k`: up to 200,000 examples.
- `HuggingFaceH4/ultrachat_200k`: up to 200,000 examples.
- Up to 200,000 prepared broad-chat conversations.
- Deterministic verified arithmetic and logic examples.
- Training splits from ARC, OpenBookQA, and BoolQ.

Reasoning records are deduplicated by normalized conversation-prefix SHA-256.
Thirteen-token benchmark n-grams are removed before the train/validation split.
The generated manifest records source revisions, counts, hashes, duplicate
counts, and contamination removals.
