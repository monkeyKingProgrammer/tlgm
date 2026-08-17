import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import load_tokenizer


class TokenBinDataset(Dataset):
    """Memory-mapped uint16 token stream split into fixed windows."""

    def __init__(self, bin_path: str | Path, context_length: int):
        self.bin_path = Path(bin_path)
        self.context_length = context_length
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if len(self.tokens) <= context_length + 1:
            raise ValueError(f"Not enough tokens in {self.bin_path}")

    def __len__(self) -> int:
        return len(self.tokens) // self.context_length - 1

    def __getitem__(self, index: int):
        start = index * self.context_length
        end = start + self.context_length + 1
        chunk = torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64))
        input_ids = chunk[:-1].long()
        labels = chunk[1:].long()
        return input_ids, labels


def read_meta(meta_path: str | Path) -> dict:
    with Path(meta_path).open("r", encoding="utf-8") as f:
        return json.load(f)


class ChatSFTDataset(Dataset):
    """JSONL chat dataset with loss only on assistant responses.

    The model still receives the full prompt text, but labels are -100 for user
    text and assistant prefixes. This is much better for chat SFT than plain LM
    training on flattened conversations.
    """

    def __init__(self, jsonl_path: str | Path, tokenizer_dir: str | Path, context_length: int):
        self.path = Path(jsonl_path)
        self.tokenizer = load_tokenizer(Path(tokenizer_dir))
        self.context_length = context_length
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def __getitem__(self, index: int):
        row = self.samples[index]
        input_ids = [self.bos_id]
        labels = [-100]
        for msg in row.get("conversations", []):
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                prefix = self._encode("Assistant: ")
                answer = self._encode(content + "\n")
                input_ids.extend(prefix)
                labels.extend([-100] * len(prefix))
                input_ids.extend(answer)
                labels.extend(answer)
            elif role == "user":
                text_ids = self._encode(f"User: {content}\n")
                input_ids.extend(text_ids)
                labels.extend([-100] * len(text_ids))
            else:
                text_ids = self._encode(f"{role.capitalize()}: {content}\n")
                input_ids.extend(text_ids)
                labels.extend([-100] * len(text_ids))
        input_ids.append(self.eos_id)
        labels.append(self.eos_id)

        input_ids = input_ids[: self.context_length]
        labels = labels[: self.context_length]
        pad_len = self.context_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)
            labels.extend([-100] * pad_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
