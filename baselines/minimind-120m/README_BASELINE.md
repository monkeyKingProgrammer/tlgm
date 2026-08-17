# MiniMind 120M Tiny ChatGPT

Fresh 120M-parameter MiniMind training project.

This project is separate from the previous 64M run. It is intended to train a stronger tiny chat model from random initialization with a much larger and more balanced 2B-token pretraining corpus.

Important constraints:

- No pretrained/base model checkpoint.
- No MiniMind pretrained weights.
- No Qwen/Llama/GPT weights.
- Pipeline: random initialization -> 2B-token pretraining -> broad SFT -> chat repair SFT -> chat CLI.
- Tokenizer: MiniMind tokenizer from `../MiniMind/model` is reused, but model weights are random.

## Model

Target: about 120M parameters.

Actual MiniMind dense config:

- Parameters: about 122.29M
- Hidden size: 960
- Layers: 10
- Attention heads: 10
- KV heads: 5
- Vocabulary: 6,400
- MoE: disabled
- Pretrain context length: 1024

Config:

```text
configs/pretrain_120m_2b_4060ti.yaml
```

## Pretraining Data Plan

Total target: 2B MiniMind-tokenizer tokens.

Revised balanced mix:

```text
40% FineWeb-Edu / SmolLM fineweb-edu-dedup: 800M tokens
20% Cosmopedia v2: 400M tokens
15% Wikipedia: 300M tokens
10% TinyStories: 200M tokens
10% FineWeb broad sample: 200M tokens
5% code text: 100M tokens
```

Reasoning:

- FineWeb-Edu remains the largest source because educational web text gives broad language and knowledge.
- Cosmopedia adds textbook/explanation style data that helps small models.
- Wikipedia adds compact factual grounding.
- TinyStories helps basic coherence but is capped at 10% to avoid story-only behavior.
- FineWeb broad sample adds non-educational web diversity while keeping the main web source educational.
- Code text adds light coding and procedural text without dominating the model.

The script writes local JSONL shards to:

```text
data/processed/pretrain_2b/
```

Each row has MiniMind pretraining format:

```jsonl
{"text":"...", "source":"fineweb_edu"}
```

## Run Order

Do not run training until the full pretraining data exists.

### 1. Environment

```powershell
cd C:\Users\ADMIN\minimind\minimind_120m_tiny_chatgpt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python check_gpu.py
```

### 2. Download/Prepare 2B Pretraining Data

This streams public datasets from Hugging Face and writes local shards.

```powershell
python scripts\prepare_pretrain_2b.py --target_tokens 2000000000 --output_dir data\processed\pretrain_2b
```

Optional compressed shards:

```powershell
python scripts\prepare_pretrain_2b.py --target_tokens 2000000000 --output_dir data\processed\pretrain_2b --compress
```

Debug-only data prep:

```powershell
python scripts\prepare_pretrain_2b.py --target_tokens 10000000 --output_dir data\tiny_debug\pretrain_10m
```

### 3. Validate Data

```powershell
python scripts\validate_data.py --pretrain_dir data\processed\pretrain_2b
```

### 4. Pretrain From Scratch

This initializes random model weights. The config has no `init_checkpoint`.

```powershell
python scripts\run_minimind_train.py --config configs\pretrain_120m_2b_4060ti.yaml
```

Resume:

```powershell
python scripts\run_minimind_train.py --config configs\pretrain_120m_2b_4060ti.yaml --resume
```

For pretraining, `--resume` is optional. The trainer automatically resumes from `checkpoints/pretrain_120m_960_2b.pth` if it exists. This is intended for daily stop/start training:

```powershell
python scripts\run_minimind_train.py --config configs\pretrain_120m_2b_4060ti.yaml
```

If the checkpoint exists, it resumes. If it does not exist, it starts from random initialization. To force-disable automatic pretraining resume:

```powershell
python scripts\run_minimind_train.py --config configs\pretrain_120m_2b_4060ti.yaml --no_auto_resume
```

Special daily launcher:

```powershell
python scripts\train_pretrain_session.py --hours 6
```

This separate launcher prints the current checkpoint summary before training starts: checkpoint found/not found, current step, cumulative tokens trained, remaining tokens, cumulative training time, estimated remaining time, last loss, last eval loss, and last LR.

Train for a fixed number of additional steps:

```powershell
python scripts\train_pretrain_session.py --steps 1000
```

Train until whichever limit is reached first:

```powershell
python scripts\train_pretrain_session.py --steps 1000 --hours 6
```

Preview without starting training:

```powershell
python scripts\train_pretrain_session.py --steps 1000 --hours 6 --dry_run
```

Expected output:

```text
checkpoints/pretrain_120m_960_2b.pth
checkpoints/pretrain_120m_960_2b.json
outputs/pretrain_120m_2b_loss.jsonl
```

Pretraining checkpoint behavior:

- The trainer overwrites `checkpoints/pretrain_120m_960_2b.pth` every 500 steps.
- Restart behavior is automatic: rerunning the same command resumes from this file if it exists.
- It does not create numbered checkpoint history.
- It does not create a separate large `*_resume.pth` file for pretraining.
- The `.pth` file is a single resumable training-state checkpoint containing model, optimizer, scaler, step, and metadata.
- The `.json` sidecar is small metadata only: cumulative GPU/training time from step 0, total token slots trained, last loss, last eval loss, last LR, step, model config, training config, data path, and checkpoint format.
- SFT and eval scripts can load this checkpoint format directly.

### 5. Prepare SFT Data

Broad instruction/chat mix:

```powershell
python scripts\prepare_sft_chat_mix.py --output data\processed\sft_chat_mix.jsonl
```

Repair set:

```powershell
python scripts\prepare_sft_repair.py --output data\processed\sft_chat_repair.jsonl
```

Validate:

```powershell
python scripts\validate_data.py
```

### 6. SFT

```powershell
python scripts\run_minimind_train.py --config configs\sft_120m_4060ti.yaml
```

Output:

```text
checkpoints/sft_120m_960_chat.pth
```

### 7. Chat Repair SFT

```powershell
python scripts\run_minimind_train.py --config configs\chat_repair_120m_4060ti.yaml
```

Final output:

```text
checkpoints/sft_120m_960_final.pth
```

### 8. Test Prompts

```powershell
python scripts\test_prompts.py --checkpoint checkpoints\sft_120m_960_final.pth
```

### 9. Chat CLI

```powershell
python chat_cli.py
```

Useful decoding options:

```powershell
python chat_cli.py --temperature 0.7 --top_p 0.9 --top_k 50 --max_new_tokens 160
```

### 10. Evaluation

```powershell
python eval_tiny_chat.py
```

Output:

```text
outputs/eval_results.md
```

## Expected Runtime

Exact runtime depends on dataset download speed and GPU clocks.

The 64M model needed about 30 minutes for roughly 65M token positions. This 120M config and 2B-token target is much larger. Expect many hours to multiple days on one RTX 4060 Ti 16GB.

## Important Limitations

This still will not truly rival a strong 0.8B model trained on hundreds of billions or trillions of tokens. The goal is to get the strongest practical tiny model from scratch on one 16GB GPU.

Quality depends more on pretraining data quality and token count than SFT. SFT can make the model chat-shaped, but it cannot add broad intelligence that pretraining did not learn.

## Files Reused From MiniMind

Reused:

- `../MiniMind/model/model_minimind.py`
- `../MiniMind/dataset/lm_dataset.py` for SFT formatting
- `../MiniMind/model` tokenizer and chat template

Not reused:

- No MiniMind pretrained model checkpoints.
- No external pretrained model checkpoints.
