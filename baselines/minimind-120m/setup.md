# Setup

Use Python 3.10 or 3.11 with PyTorch CUDA.

Windows PowerShell:

```powershell
cd C:\Users\ADMIN\minimind\minimind_120m_tiny_chatgpt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python check_gpu.py
```

If PyTorch CPU-only gets installed, reinstall PyTorch from the official CUDA wheel index for your CUDA version.

Storage expectation:

- 2B-token processed JSONL may require hundreds of GB depending on text lengths and compression.
- Use `--compress` in `prepare_pretrain_2b.py` if disk is tight, but loading gzip shards can be slower.
- Keep at least 300GB free for data, checkpoints, resume checkpoints, and logs.

RTX 4060 Ti 16GB safe defaults:

- Pretrain: context 1024, micro batch 2, gradient accumulation 32.
- SFT: context 1024, micro batch 1, gradient accumulation 32.
- If CUDA OOM occurs, reduce `batch_size` first, then `context_length`.
