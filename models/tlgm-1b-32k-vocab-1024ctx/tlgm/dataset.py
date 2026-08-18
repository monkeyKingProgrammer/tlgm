import json
import math
import random
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .tokenizer import load_tokenizer


class TokenBinDataset(Dataset):
    """Memory-mapped uint16 token stream split into fixed windows."""

    def __init__(self, bin_path: str | Path, context_length: int):
        self.bin_path = Path(bin_path)
        self.context_length = context_length
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        if len(self.tokens) < context_length:
            raise ValueError(f"Not enough tokens in {self.bin_path}")

    def __len__(self) -> int:
        return len(self.tokens) // self.context_length

    def __getitem__(self, index: int):
        start = index * self.context_length
        end = start + self.context_length
        chunk = torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64))
        input_ids = chunk.long()
        # The model performs the one-token shift in its loss function.
        labels = input_ids.clone()
        return input_ids, labels

    def close(self) -> None:
        mmap = getattr(getattr(self, "tokens", None), "_mmap", None)
        if mmap is not None:
            mmap.close()

    def __del__(self):
        self.close()


class EvenHoldoutTrainDataset(Dataset):
    """Dataset view excluding holdout indices spread across the full corpus."""

    def __init__(self, dataset: Dataset, holdout_indices: list[int]):
        self.dataset = dataset
        self.holdout_indices = sorted(holdout_indices)

    def __len__(self) -> int:
        return len(self.dataset) - len(self.holdout_indices)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        low = index
        high = index + len(self.holdout_indices)
        while low < high:
            candidate = (low + high) // 2
            non_holdout_through_candidate = candidate + 1 - bisect_right(self.holdout_indices, candidate)
            if non_holdout_through_candidate <= index:
                low = candidate + 1
            else:
                high = candidate
        return self.dataset[low]


def even_holdout_indices(dataset_size: int, holdout_size: int) -> list[int]:
    """Return unique midpoint-stratified indices spanning the entire dataset."""
    if not 0 < holdout_size < dataset_size:
        raise ValueError("holdout_size must be between zero and dataset_size")
    return [((2 * index + 1) * dataset_size) // (2 * holdout_size) for index in range(holdout_size)]


class ResumableRandomSampler(Sampler[int]):
    """Deterministic O(1)-memory permutation with checkpointable position."""

    def __init__(self, data_source: Dataset, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.offset = 0

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.offset)

    def _permutation_parameters(self) -> tuple[int, int]:
        size = len(self.data_source)
        if size == 1:
            return 1, 0
        rng = random.Random(self.seed + self.epoch)
        multiplier = rng.randrange(1, size)
        while math.gcd(multiplier, size) != 1:
            multiplier = (multiplier + 1) % size
            if multiplier == 0:
                multiplier = 1
        return multiplier, rng.randrange(size)

    def __iter__(self):
        size = len(self.data_source)
        if size == 0:
            return
        multiplier, increment = self._permutation_parameters()
        position = self.offset
        while position < size:
            self.offset = position + 1
            yield (multiplier * position + increment) % size
            position += 1
        self.epoch += 1
        self.offset = 0

    def state_dict(self) -> dict:
        return {"seed": self.seed, "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, state: dict) -> None:
        if not state:
            return
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("Sampler seed does not match checkpoint seed")
        self.epoch = int(state.get("epoch", 0))
        self.offset = int(state.get("offset", 0))
        if not 0 <= self.offset <= len(self.data_source):
            raise ValueError("Sampler checkpoint offset is outside the dataset")


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
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    row = json.loads(line)
                    messages = row.get("conversations", [])
                    configured_target = row.get("target_index")
                    target_indices = (
                        [int(configured_target)]
                        if configured_target is not None
                        else range(len(messages))
                    )
                    for target_index in target_indices:
                        if not 0 <= target_index < len(messages):
                            raise ValueError(f"Invalid target_index at byte offset {offset} in {self.path}")
                        message = messages[target_index]
                        if message.get("role") == "assistant" and str(message.get("content", "")).strip():
                            self.samples.append((offset, target_index))
        if not self.samples:
            raise ValueError(f"No conversations with assistant responses found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_row(self, index: int) -> dict:
        offset, _ = self.samples[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            return json.loads(handle.readline().decode("utf-8"))

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def _truncate(self, input_ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
        if len(input_ids) <= self.context_length:
            return input_ids, labels

        target_start = next((i for i, label in enumerate(labels) if label != -100), len(labels))
        available_prompt = max(0, target_start - 1)
        answer_length = len(input_ids) - target_start
        minimum_prompt = min(available_prompt, self.context_length // 3)
        prompt_budget = min(
            available_prompt,
            max(minimum_prompt, self.context_length - answer_length - 1),
        )
        answer_budget = self.context_length - prompt_budget - 1
        prompt_start = max(1, target_start - prompt_budget)
        answer_ids = input_ids[target_start:]
        answer_labels = labels[target_start:]
        if len(answer_ids) > answer_budget:
            separator = self._encode("\n...\n")
            if answer_budget >= len(separator) + 2:
                tail_budget = max(1, min(max(16, answer_budget // 4), answer_budget // 2))
                prefix_budget = max(1, answer_budget - tail_budget - len(separator))
                tail_budget = max(1, answer_budget - prefix_budget - len(separator))
                answer_ids = answer_ids[:prefix_budget] + separator + answer_ids[-tail_budget:]
                answer_labels = answer_labels[:prefix_budget] + separator + answer_labels[-tail_budget:]
            else:
                # Preserve EOS/final-answer tokens even at extremely small contexts.
                prefix_budget = max(0, answer_budget - 1)
                answer_ids = answer_ids[:prefix_budget] + answer_ids[-1:]
                answer_labels = answer_labels[:prefix_budget] + answer_labels[-1:]
        kept_ids = [self.bos_id] + input_ids[prompt_start:target_start] + answer_ids
        kept_labels = [-100] * (1 + target_start - prompt_start) + answer_labels
        return kept_ids, kept_labels

    def __getitem__(self, index: int):
        row = self._read_row(index)
        _, target_index = self.samples[index]
        input_ids = [self.bos_id]
        labels = [-100]
        for message_index, msg in enumerate(row.get("conversations", [])):
            if message_index > target_index:
                break
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
                labels.extend(answer if message_index == target_index else [-100] * len(answer))
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

        input_ids, labels = self._truncate(input_ids, labels)
        pad_len = self.context_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_id] * pad_len)
            labels.extend([-100] * pad_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
