import json
import math
import pickle
import random
import signal
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TLGMConfig
from .dataset import (
    ChatSFTDataset,
    EvenHoldoutTrainDataset,
    ResumableRandomSampler,
    TokenBinDataset,
    even_holdout_indices,
    read_meta,
)
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
        self.stop_requested = False
        self.stop_signal = None
        self.session_id = uuid.uuid4().hex
        self.validate_training_config()

    def validate_training_config(self) -> None:
        required_positive = (
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_steps",
            "save_steps",
            "grad_clip",
        )
        for name in required_positive:
            if float(self.train_cfg.get(name, 0)) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if int(self.train_cfg.get("warmup_steps", 0)) < 0:
            raise ValueError("training.warmup_steps cannot be negative")
        if self.train_cfg.get("dtype") not in {"float32", "float16", "bfloat16"}:
            raise ValueError("training.dtype must be float32, float16, or bfloat16")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested, but CUDA is not available")

    def resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.project_dir / p

    def load_optimizer_state(self, optimizer, saved_state: dict, model) -> None:
        saved_groups = saved_state.get("param_groups", [])
        saved_ids = [param_id for group in saved_groups for param_id in group.get("params", [])]
        current_params = list(model.parameters())
        if len(saved_ids) != len(current_params):
            raise ValueError(
                f"Optimizer checkpoint has {len(saved_ids)} parameters; model has {len(current_params)}"
            )
        optimizer.state.clear()
        for parameter, saved_id in zip(current_params, saved_ids):
            state = saved_state.get("state", {}).get(saved_id)
            if state:
                optimizer.state[parameter] = state

    def checkpoint_to_load(self) -> Path | None:
        if self.checkpoint_path.exists():
            return self.checkpoint_path
        previous = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".previous")
        if previous.exists():
            print(f"Primary checkpoint missing. Recovering from previous checkpoint: {previous}")
            return previous
        return None

    def load_checkpoint(self, model, optimizer, scaler):
        resume_path = self.checkpoint_to_load()
        if resume_path is None:
            init_checkpoint = self.raw_config.get("init_checkpoint")
            if init_checkpoint:
                init_path = self.resolve(init_checkpoint)
                print(f"No checkpoint found. Loading init checkpoint: {init_path}")
                payload = torch.load(init_path, map_location=self.device, weights_only=False)
                model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload, strict=True)
                return self.empty_resume_state()
            print("No checkpoint found. Starting from random initialization.")
            return self.empty_resume_state()
        print(f"Checkpoint found. Resuming from: {resume_path}")
        try:
            payload = torch.load(resume_path, map_location=self.device, weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            previous = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".previous")
            if resume_path == self.checkpoint_path and previous.exists():
                print(f"Primary checkpoint failed integrity/load check ({error}). Recovering from: {previous}")
                resume_path = previous
                payload = torch.load(resume_path, map_location=self.device, weights_only=False)
            else:
                raise
        model.load_state_dict(payload["model"], strict=True)
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except ValueError:
            self.load_optimizer_state(optimizer, payload["optimizer"], model)
            print("Migrated legacy single-group optimizer state into decay/no-decay groups.")
        scaler.load_state_dict(payload["scaler"])
        meta = payload.get("metadata", {})
        self.print_resume_summary(meta)
        checkpoint_step = int(payload.get("global_step", meta.get("step", 0)))
        return {
            "global_step": checkpoint_step,
            "cumulative_seconds": float(meta.get("cumulative_gpu_time_seconds", 0.0)),
            "seen_tokens": int(meta.get("total_seen_tokens", meta.get("total_tokens_trained", 0))),
            "supervised_tokens": int(meta.get("total_supervised_tokens", 0)),
            "nonpad_tokens": int(meta.get("total_nonpad_input_tokens", 0)),
            "nonpad_counter_start_step": int(meta.get("nonpad_counter_start_step", checkpoint_step)),
            "sampler_state": payload.get("sampler_state", {}),
            "rng_state": payload.get("rng_state", {}),
            "data_fingerprint": meta.get("data_fingerprint"),
        }

    @staticmethod
    def empty_resume_state() -> dict:
        return {
            "global_step": 0,
            "cumulative_seconds": 0.0,
            "seen_tokens": 0,
            "supervised_tokens": 0,
            "nonpad_tokens": 0,
            "nonpad_counter_start_step": 0,
            "sampler_state": {},
            "rng_state": {},
            "data_fingerprint": None,
        }

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

    @staticmethod
    def capture_rng_state() -> dict:
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    @staticmethod
    def restore_rng_state(state: dict) -> None:
        if not state:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        # Checkpoint loading maps every tensor to the training device. RNG APIs
        # specifically require CPU ByteTensors, even for saved CUDA states.
        torch_cpu = torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu").contiguous()
        torch.set_rng_state(torch_cpu)
        if torch.cuda.is_available() and state.get("torch_cuda"):
            torch_cuda = [
                torch.as_tensor(item, dtype=torch.uint8, device="cpu").contiguous()
                for item in state["torch_cuda"]
            ]
            torch.cuda.set_rng_state_all(torch_cuda)

    @staticmethod
    def state_dict_for_save(model, dtype_name: str) -> dict:
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype_name not in dtypes:
            raise ValueError(f"Unsupported checkpoint model dtype: {dtype_name}")
        dtype = dtypes[dtype_name]
        return {
            key: value.detach().to(device="cpu", dtype=dtype) if value.is_floating_point() else value.detach().cpu()
            for key, value in model.state_dict().items()
        }

    @staticmethod
    def optimizer_param_names(model, optimizer) -> list[list[str]]:
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        return [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]

    def save(self, model, optimizer, scaler, sampler, step: int, metadata: dict) -> None:
        model_dtype = str(self.train_cfg.get("checkpoint_model_dtype", "float32"))
        payload = {
            "checkpoint_schema_version": 2,
            "model": self.state_dict_for_save(model, model_dtype),
            "optimizer": optimizer.state_dict(),
            "optimizer_param_names": self.optimizer_param_names(model, optimizer),
            "scaler": scaler.state_dict(),
            "sampler_state": sampler.state_dict(),
            "rng_state": self.capture_rng_state(),
            "global_step": step,
            "config": self.raw_config,
            "metadata": metadata,
        }
        atomic_torch_save(payload, self.checkpoint_path, keep_previous=True)
        atomic_json_save(metadata, self.meta_path)

    def save_best_model(self, model, path: Path, metadata: dict) -> None:
        model_dtype = str(self.train_cfg.get("best_model_dtype", "bfloat16"))
        payload = {
            "checkpoint_schema_version": 2,
            "model": self.state_dict_for_save(model, model_dtype),
            "config": self.raw_config,
            "metadata": metadata,
        }
        atomic_torch_save(payload, path)
        atomic_json_save(metadata, path.with_suffix(".json"))

    def build_optimizer(self, model):
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        common = {
            "lr": float(self.train_cfg["learning_rate"]),
        }
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(self.train_cfg.get("weight_decay", 0.1))},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            **common,
        )

    def install_signal_handlers(self) -> None:
        def request_stop(signum, _frame):
            self.stop_requested = True
            self.stop_signal = signal.Signals(signum).name
            print(f"\nReceived {self.stop_signal}; checkpointing after the current optimizer step.")

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

    def validate_pretrain_data(self, data_cfg: dict) -> None:
        if not data_cfg.get("meta"):
            return
        bin_path = self.resolve(data_cfg["train_bin"])
        meta = read_meta(self.resolve(data_cfg["meta"]))
        tokens = int(meta.get("tokens", -1))
        target = int(meta.get("target_tokens", tokens))
        if tokens <= 0 or tokens != target:
            raise ValueError(f"Pretraining metadata is incomplete: tokens={tokens}, target={target}")
        expected_bytes = tokens * np.dtype(np.uint16).itemsize
        actual_bytes = bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(f"Token binary size mismatch: expected {expected_bytes}, got {actual_bytes}")

    def split_dataset(self, dataset):
        validation_cfg = self.raw_config.get("validation", {})
        if not validation_cfg.get("enabled", False):
            return dataset, None
        if len(dataset) < 2:
            raise ValueError("Validation requires at least two dataset samples")
        holdout = int(validation_cfg.get("holdout_samples", 0))
        if holdout <= 0:
            holdout = max(1, min(10_000, len(dataset) // 100))
        holdout = min(holdout, len(dataset) - 1)
        holdout_indices = even_holdout_indices(len(dataset), holdout)
        return EvenHoldoutTrainDataset(dataset, holdout_indices), Subset(dataset, holdout_indices)

    def data_fingerprint(self, data_cfg: dict) -> str | None:
        manifest_name = data_cfg.get("manifest")
        if not manifest_name:
            return None
        manifest_path = self.resolve(manifest_name)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        return manifest.get("train", {}).get("sha256") or manifest.get("sha256")

    def evaluate(self, model, loader, amp_dtype, max_batches: int) -> tuple[float, float, int]:
        model.eval()
        weighted_loss = 0.0
        supervised_tokens = 0
        with torch.inference_mode():
            for batch_index, (input_ids, labels) in enumerate(loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                target_count = int((labels[:, 1:] != -100).sum().item())
                if target_count == 0:
                    continue
                eval_ctx = (
                    torch.amp.autocast("cuda", dtype=amp_dtype)
                    if amp_dtype is not None and "cuda" in self.device
                    else nullcontext()
                )
                with eval_ctx:
                    loss = model(input_ids, labels)["loss"]
                if torch.isfinite(loss):
                    weighted_loss += float(loss.item()) * target_count
                    supervised_tokens += target_count
        model.train()
        if supervised_tokens == 0:
            return float("nan"), float("nan"), 0
        mean_loss = weighted_loss / supervised_tokens
        return mean_loss, math.exp(min(mean_loss, 20.0)), supervised_tokens

    def train(self, stop_after_seconds: float | None = None) -> None:
        self.install_signal_handlers()
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
            self.validate_pretrain_data(data_cfg)
            dataset = TokenBinDataset(self.resolve(data_cfg["train_bin"]), self.model_cfg.context_length)
        if data_cfg.get("validation_jsonl"):
            train_dataset = dataset
            validation_dataset = ChatSFTDataset(
                self.resolve(data_cfg["validation_jsonl"]),
                self.resolve(data_cfg.get("tokenizer_dir", "tokenizer")),
                self.model_cfg.context_length,
            )
        else:
            train_dataset, validation_dataset = self.split_dataset(dataset)
        optimizer = self.build_optimizer(model)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.train_cfg["dtype"] == "float16" and "cuda" in self.device,
        )
        resume = self.load_checkpoint(model, optimizer, scaler)
        current_data_fingerprint = self.data_fingerprint(data_cfg)
        sampler = ResumableRandomSampler(train_dataset, int(self.train_cfg.get("seed", 120)))
        if resume["global_step"] and resume["data_fingerprint"] != current_data_fingerprint:
            print("Training-data fingerprint changed or was absent; resetting only the sampler position.")
        else:
            sampler.load_state_dict(resume["sampler_state"])
        loader = DataLoader(
            train_dataset,
            batch_size=int(self.train_cfg["batch_size"]),
            sampler=sampler,
            num_workers=int(self.train_cfg.get("num_workers", 0)),
            pin_memory="cuda" in self.device,
            drop_last=True,
        )
        validation_cfg = self.raw_config.get("validation", {})
        validation_loader = None
        if validation_dataset is not None:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(validation_cfg.get("batch_size", self.train_cfg["batch_size"])),
                shuffle=False,
                num_workers=int(self.train_cfg.get("num_workers", 0)),
                pin_memory="cuda" in self.device,
                drop_last=False,
            )
        global_step = resume["global_step"]
        cumulative_seconds = resume["cumulative_seconds"]
        total_seen_tokens = resume["seen_tokens"]
        total_supervised_tokens = resume["supervised_tokens"]
        total_nonpad_input_tokens = resume["nonpad_tokens"]
        nonpad_counter_start_step = resume["nonpad_counter_start_step"]
        dtype = self.train_cfg["dtype"]
        amp_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else None
        max_steps = int(self.train_cfg["max_steps"])
        accum = int(self.train_cfg["gradient_accumulation_steps"])
        save_steps = int(self.train_cfg["save_steps"])
        warmup_steps = int(self.train_cfg["warmup_steps"])
        base_lr = float(self.train_cfg["learning_rate"])
        tokens_per_step = int(self.train_cfg["batch_size"]) * accum * self.model_cfg.context_length
        iterator = iter(loader)
        self.restore_rng_state(resume["rng_state"])
        model.train()
        if "cuda" in self.device:
            torch.cuda.synchronize()
        run_start = time.perf_counter()
        progress = tqdm(total=max_steps, initial=global_step, desc="tlgm-pretrain")
        session_record = {
            "type": "session_start",
            "session_id": self.session_id,
            "step": global_step,
            "checkpoint": str(self.checkpoint_path),
            "started_unix": time.time(),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(session_record) + "\n")
        best_checkpoint = self.raw_config.get("best_checkpoint_path")
        best_checkpoint_path = self.resolve(best_checkpoint) if best_checkpoint else None
        best_validation_loss = float("inf")
        if best_checkpoint_path and best_checkpoint_path.with_suffix(".json").exists():
            with best_checkpoint_path.with_suffix(".json").open("r", encoding="utf-8") as handle:
                best_validation_loss = float(json.load(handle).get("validation_loss", float("inf")))
        consecutive_invalid_steps = 0
        max_invalid_steps = int(self.train_cfg.get("max_consecutive_invalid_steps", 10))
        while global_step < max_steps:
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            finite = 0
            step_seen_tokens = 0
            step_supervised_tokens = 0
            step_nonpad_input_tokens = 0
            invalid_step = False
            for _ in range(accum):
                try:
                    input_ids, labels = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    input_ids, labels = next(iterator)
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None and "cuda" in self.device else nullcontext()
                with ctx:
                    loss = model(input_ids, labels)["loss"] / accum
                if not torch.isfinite(loss):
                    invalid_step = True
                    break
                scaler.scale(loss).backward()
                running_loss += float(loss.item()) * accum
                finite += 1
                step_seen_tokens += input_ids.numel()
                step_supervised_tokens += int((labels[:, 1:] != -100).sum().item())
                if data_cfg.get("sft_jsonl"):
                    step_nonpad_input_tokens += int((input_ids != self.model_cfg.pad_token_id).sum().item())
                else:
                    step_nonpad_input_tokens += input_ids.numel()
            if invalid_step or finite != accum:
                optimizer.zero_grad(set_to_none=True)
                consecutive_invalid_steps += 1
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "type": "invalid_step",
                        "step": global_step,
                        "consecutive": consecutive_invalid_steps,
                    }) + "\n")
                if scaler.is_enabled():
                    scaler.update(new_scale=max(float(scaler.get_scale()) / 2.0, 1.0))
                if consecutive_invalid_steps >= max_invalid_steps:
                    raise FloatingPointError(
                        f"Aborting after {consecutive_invalid_steps} consecutive non-finite accumulation attempts"
                    )
                continue
            consecutive_invalid_steps = 0
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
            total_seen_tokens += step_seen_tokens
            total_supervised_tokens += step_supervised_tokens
            total_nonpad_input_tokens += step_nonpad_input_tokens
            record = {
                "type": "train",
                "step": global_step,
                "loss": loss_value,
                "lr": lr,
                "total_seen_tokens": total_seen_tokens,
                "total_supervised_tokens": total_supervised_tokens,
                "total_nonpad_input_tokens": total_nonpad_input_tokens,
                "nonpad_counter_start_step": nonpad_counter_start_step,
                "cumulative_gpu_time_seconds": total_seconds,
            }
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.writer.add_scalar("train/loss", loss_value, global_step)
            self.writer.add_scalar("train/lr", lr, global_step)
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{lr:.2e}")
            progress.update(1)
            eval_steps = int(validation_cfg.get("eval_steps", 0))
            if validation_loader is not None and eval_steps and global_step % eval_steps == 0:
                val_loss, perplexity, val_tokens = self.evaluate(
                    model,
                    validation_loader,
                    amp_dtype,
                    int(validation_cfg.get("eval_batches", 0)),
                )
                validation_record = {
                    "type": "validation",
                    "step": global_step,
                    "loss": val_loss,
                    "perplexity": perplexity,
                    "supervised_tokens": val_tokens,
                }
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(validation_record) + "\n")
                self.writer.add_scalar("validation/loss", val_loss, global_step)
                self.writer.add_scalar("validation/perplexity", perplexity, global_step)
                print(f"\nValidation step {global_step}: loss={val_loss:.4f}, perplexity={perplexity:.3f}")
                if best_checkpoint_path and math.isfinite(val_loss) and val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    best_metadata = {
                        "step": global_step,
                        "validation_loss": val_loss,
                        "validation_perplexity": perplexity,
                        "total_seen_tokens": total_seen_tokens,
                        "cumulative_gpu_time_seconds": total_seconds,
                    }
                    self.save_best_model(model, best_checkpoint_path, best_metadata)
                    print(f"Saved best validation model: {best_checkpoint_path}")
            should_save = (
                global_step % save_steps == 0
                or global_step == max_steps
                or self.stop_requested
                or (stop_after_seconds and elapsed >= stop_after_seconds)
            )
            if should_save:
                metadata = {
                    "step": global_step,
                    "max_steps": max_steps,
                    "last_loss": loss_value,
                    "last_lr": lr,
                    "tokens_per_step": tokens_per_step,
                    "total_tokens_trained": total_seen_tokens,
                    "total_seen_tokens": total_seen_tokens,
                    "total_supervised_tokens": total_supervised_tokens,
                    "total_context_positions": total_seen_tokens,
                    "total_nonpad_input_tokens": total_nonpad_input_tokens,
                    "nonpad_counter_start_step": nonpad_counter_start_step,
                    "cumulative_gpu_time_seconds": total_seconds,
                    "cumulative_gpu_time_hours": total_seconds / 3600,
                    "architecture": "TLGM lower-triangular causal sequence mixer",
                    "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
                    "checkpoint_model_dtype": str(self.train_cfg.get("checkpoint_model_dtype", "float32")),
                    "session_id": self.session_id,
                    "stop_signal": self.stop_signal,
                    "data_fingerprint": current_data_fingerprint,
                }
                self.save(model, optimizer, scaler, sampler, global_step, metadata)
                print(f"\nSaved TLGM checkpoint at step {global_step}: {self.checkpoint_path}")
            if self.stop_requested or (stop_after_seconds and elapsed >= stop_after_seconds):
                reason = self.stop_signal or "session time limit"
                print(f"Stopping after checkpoint: {reason}.")
                break
        progress.close()
        self.writer.close()
