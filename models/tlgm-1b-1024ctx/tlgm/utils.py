import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if warmup_steps and step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def atomic_torch_save(payload, path: Path, keep_previous: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    with tmp.open("r+b") as handle:
        os.fsync(handle.fileno())
    if keep_previous and path.exists():
        previous = path.with_suffix(path.suffix + ".previous")
        os.replace(path, previous)
    os.replace(tmp, path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
