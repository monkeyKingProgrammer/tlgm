# Model Weights And Checkpoints

## Why Weights Are Not Stored In Git

GitHub rejects individual Git objects larger than 100 MB. Full training
checkpoints in this project contain model weights, AdamW moments, gradients or
scaler state, sampler state, and random state; they range from roughly 1.1 GB
for 120M experiments to more than 12 GB for the 1B trainer checkpoint.

Committing such files would make the repository unusable even with Git LFS.
Versioned model-only exports should therefore be attached to GitHub Releases
in shards small enough for GitHub's release-asset limit.

## Planned Release Assets

| Release | Source checkpoint | Export format |
|---|---|---|
| `tlgm-120m-final` | `tlgm_120m_sft_polished.pth` | model-only weights + config + tokenizer |
| `tlgm-120m-1024ctx-final` | `tlgm_120m_1024ctx_sft_polished.pth` | model-only weights + config + tokenizer |
| `tlgm-1b-1024ctx-reasoning-final` | `tlgm_1b_1024ctx_sft_reasoning_3day.pth` | sharded BF16 model-only weights + manifest |

The full optimizer checkpoints remain local archival artifacts. They are
needed only to resume training, not for inference.

Each public weight release should include:

- Model-only tensors.
- Architecture YAML.
- Tokenizer vocabulary and merges.
- SHA-256 manifest.
- Source training-step metadata.
- Precision and parameter count.
- Loading example.
- Link to [MODEL_CARD.md](MODEL_CARD.md).

## Local Checkpoint Paths

After downloading, place assets under the corresponding model directory:

```text
models/tlgm-120m/checkpoints/
models/tlgm-120m-1024ctx/checkpoints/
models/tlgm-1b-1024ctx/checkpoints/
```

These directories are ignored by Git.

## Security

PyTorch pickle checkpoints can execute code during deserialization. Only load
`.pth` files from trusted sources and verify SHA-256 hashes. Model-only
Safetensors exports are preferred for public distribution because they avoid
pickle execution.
