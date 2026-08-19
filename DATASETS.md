# Dataset Lineage And Preparation

This repository does not redistribute raw upstream datasets or generated
token binaries. Data-preparation scripts download or stream upstream sources,
filter rows, tokenize documents, and produce local training artifacts.

Users are responsible for reviewing the current dataset cards, licenses,
terms, privacy considerations, and allowed uses before downloading data.

## TLGM-120M And TLGM-120M-1024

The first 2B-token stream was derived from the public-text mix prepared for
the MiniMind 120M baseline. The preparation pipeline targeted TinyStories,
English Wikipedia, FineWeb-Edu/OpenWeb-style educational web text, and related
clean public sources according to the scripts in the baseline folder.

TLGM did not reuse MiniMind model weights. It trained its own byte-level BPE
tokenizer and converted the prepared JSONL documents into its own contiguous
`uint16` token stream.

The 1024-context 120M model first trained on 2B tokens, then continued on a
3B-token knowledge stream:

| Source | Weight | Tokens |
|---|---:|---:|
| `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 40% | 1.2B |
| `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 30% | 0.9B |
| `wikimedia/wikipedia`, `20231101.en` | 30% | 0.9B |

## TLGM-1B Pretraining

The 1B corpus builder creates exactly 20,000,000,000 token IDs:

| Source | Weight | Token quota |
|---|---:|---:|
| `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 40% | 8B |
| `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 25% | 5B |
| `wikimedia/wikipedia`, `20231101.en` | 20% | 4B |
| `HuggingFaceFW/fineweb-edu`, `sample-10BT` | 15% | 3B |

The generated binary is exactly 40,000,000,000 bytes because every ID is
stored as little-endian `uint16`. Documents receive `<bos>` and `<eos>`
boundaries. A fixed held-out tail is reserved for validation.

Preparation:

```bash
cd models/tlgm-1b-1024ctx
python3 scripts/prepare_pretrain20b_tokens.py
python3 scripts/validate_pretrain_data.py
```

The builder supports interrupted download/tokenization recovery through a
progress manifest. Training memory-maps the result rather than loading 40 GB
into RAM.

## Supervised Fine-Tuning

SFT rows use this schema:

```json
{
  "conversations": [
    {"role": "user", "content": "What is 2+3?"},
    {"role": "assistant", "content": "2+3 is 5."}
  ]
}
```

The renderer creates a `<bos> User: ... Assistant: ... <eos>` sequence. User
text, role prefixes, padding, and BOS are assigned label `-100`; only assistant
answer tokens and EOS contribute to the loss.

The post-training mix includes public instruction and reasoning sources plus
project-authored examples for greetings, factual questions, explanations,
stories, arithmetic, uncertainty, correction behavior, and short chat. Exact
builders, source identifiers, filtering, balancing, and manifests are in:

```text
models/tlgm-1b-1024ctx/scripts/
models/tlgm-1b-1024ctx/configs/
```

Because public datasets can change, reproducible runs should pin dataset
revisions and archive generated manifests containing source names, row counts,
token counts, hashes, and filtering settings.

## Data Exclusions

The Git repository intentionally excludes:

- Raw dataset downloads.
- Tokenized `.bin` streams.
- Large JSONL corpora.
- Hugging Face caches.
- Benchmark dataset caches.
- Any credentials or private network configuration.
