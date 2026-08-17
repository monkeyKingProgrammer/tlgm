import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TLGMConfig
from .dataset import ChatSFTDataset, TokenBinDataset
from .model import TLGMForCausalLM
from .utils import atomic_json_save, atomic_torch_save, cosine_lr, seed_everything


class TLGMTrainer:
    def __init__(self, config: dict, project_dir: Path):
        self.raw_config = config
        self.project_dir = project_dir
        self.train_cfg = config["training"]
        self.model_cfg = TLGMConfig.from_dict(config["model"])
        self.device = self.train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            self.device = "cuda:0"
        seed_everything(int(self.train_cfg.get("seed", 120)))
        self.checkpoint_path = self.resolve(config["checkpoint_path"])
        self.meta_path = self.checkpoint_path.with_suffix(".json")
        self.log_path = self.resolve(config["log_path"])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.resolve(config.get("tensorboard_dir", "outputs/tensorboard"))))

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.project_dir / p

    def load_checkpoint(self, model, optimizer, scaler):
        if not self.checkpoint_path.exists():
            init_checkpoint = self.raw_config.get("init_checkpoint")
            if init_checkpoint:
                init_path = self.resolve(init_checkpoint)
                print(f"No checkpoint found. Loading init checkpoint: {init_path}")
                payload = torch.load(init_path, map_location=self.device)
                model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
                return 0, 0.0
            print("No checkpoint found. Starting from random initialization.")
            return 0, 0.0
        print(f"Checkpoint found. Resuming from: {self.checkpoint_path}")
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        meta = payload.get("metadata", {})
        self.print_resume_summary(meta)
        return int(payload.get("global_step", meta.get("step", 0))), float(meta.get("cumulative_gpu_time_seconds", 0.0))

    @staticmethod
    def print_resume_summary(meta: dict) -> None:
        print("=" * 72)
        print("TLGM checkpoint summary")
        print(f"Step: {meta.get('step', 'unknown')}")
        print(f"Total tokens trained: {meta.get('total_tokens_trained', 'unknown')}")
        print(f"Cumulative time hours: {meta.get('cumulative_gpu_time_hours', 'unknown')}")
        print(f"Last loss: {meta.get('last_loss', 'unknown')}")
        print(f"Last lr: {meta.get('last_lr', 'unknown')}")
        print("=" * 72)

    def save(self, model, optimizer, scaler, step: int, metadata: dict) -> None:
        payload = {
            "model": {k: v.detach().half().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "global_step": step,
            "config": self.raw_config,
            "metadata": metadata,
        }
        atomic_torch_save(payload, self.checkpoint_path)
        atomic_json_save(metadata, self.meta_path)

    def train(self, stop_after_seconds: float | None = None) -> None:
        model = TLGMForCausalLM(self.model_cfg).to(self.device)
        print(json.dumps(model.parameter_report(), indent=2))
        data_cfg = self.raw_config["data"]
        if data_cfg.get("sft_jsonl"):
            dataset = ChatSFTDataset(
                self.resolve(data_cfg["sft_jsonl"]),
                self.resolve(data_cfg.get("tokenizer_dir", "tokenizer")),
                self.model_cfg.context_length,
            )
        else:
            dataset = TokenBinDataset(self.resolve(data_cfg["train_bin"]), self.model_cfg.context_length)
        loader = DataLoader(
            dataset,
            batch_size=int(self.train_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            pin_memory="cuda" in self.device,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.train_cfg["learning_rate"]), weight_decay=float(self.train_cfg.get("weight_decay", 0.1)))
        scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg["dtype"] == "float16" and "cuda" in self.device)
        global_step, cumulative_seconds = self.load_checkpoint(model, optimizer, scaler)
        dtype = self.train_cfg["dtype"]
        amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else None
        max_steps = int(self.train_cfg["max_steps"])
        accum = int(self.train_cfg["gradient_accumulation_steps"])
        save_steps = int(self.train_cfg["save_steps"])
        warmup_steps = int(self.train_cfg["warmup_steps"])
        base_lr = float(self.train_cfg["learning_rate"])
        tokens_per_step = int(self.train_cfg["batch_size"]) * accum * self.model_cfg.context_length
        iterator = iter(loader)
        model.train()
        if "cuda" in self.device:
            torch.cuda.synchronize()
        run_start = time.perf_counter()
        progress = tqdm(total=max_steps, initial=global_step, desc="tlgm-pretrain")
        while global_step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            finite = 0
            for _ in range(accum):
                try:
                    input_ids, labels = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    input_ids, labels = next(iterator)
                input_ids, labels = input_ids.to(self.device), labels.to(self.device)
                ctx = torch.cuda.amp.autocast(dtype=amp_dtype) if amp_dtype is not None and "cuda" in self.device else nullcontext()
                with ctx:
                    loss = model(input_ids, labels)["loss"] / accum
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                running_loss += float(loss.item()) * accum
                finite += 1
            if finite == 0:
                continue
            global_step += 1
            lr = cosine_lr(global_step, max_steps, base_lr, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            if "cuda" in self.device:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - run_start
            total_seconds = cumulative_seconds + elapsed
            loss_value = running_loss / finite
            total_tokens = global_step * tokens_per_step
            record = {"step": global_step, "loss": loss_value, "lr": lr, "total_tokens_trained": total_tokens, "cumulative_gpu_time_seconds": total_seconds}
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.writer.add_scalar("train/loss", loss_value, global_step)
            self.writer.add_scalar("train/lr", lr, global_step)
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")
            progress.update(1)
            should_save = global_step % save_steps == 0 or global_step == max_steps or (stop_after_seconds and elapsed >= stop_after_seconds)
            if should_save:
                metadata = {
                    "step": global_step,
                    "max_steps": max_steps,
                    "last_loss": loss_value,
                    "last_lr": lr,
                    "tokens_per_step": tokens_per_step,
                    "total_tokens_trained": total_tokens,
                    "cumulative_gpu_time_seconds": total_seconds,
                    "cumulative_gpu_time_hours": total_seconds / 3600,
                    "architecture": "TLGM lower-triangular causal sequence mixer",
                }
                self.save(model, optimizer, scaler, global_step, metadata)
                print(f"\nSaved TLGM checkpoint at step {global_step}: {self.checkpoint_path}")
            if stop_after_seconds and elapsed >= stop_after_seconds:
                print("Session time limit reached.")
                break
        progress.close()
        self.writer.close()
