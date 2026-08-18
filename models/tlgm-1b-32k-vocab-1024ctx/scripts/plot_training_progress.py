import argparse
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


PROJECT_DIR = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_DIR / candidate


def load_records(path: Path):
    training = {}
    validation = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            raw = raw.replace("\x00", "").strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            if not isinstance(step, int):
                continue
            loss = float(row.get("loss", math.nan))
            if not math.isfinite(loss):
                continue
            if row.get("type") == "train":
                training[step] = row
            elif row.get("type") == "validation":
                validation[step] = row
    train = [training[step] for step in sorted(training)]
    valid = [validation[step] for step in sorted(validation)]
    if not train:
        raise ValueError(f"No valid training records found in {path}")
    return train, valid


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full(len(values), float(values.mean()))
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    means = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), means))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TLGM training progress from its JSONL log.")
    parser.add_argument("--log", default="outputs/tlgm_1b_32k_sft_reasoning_loss.jsonl")
    parser.add_argument("--output", default="outputs/training_progress.png")
    parser.add_argument("--summary", default="outputs/training_progress.json")
    parser.add_argument("--max_steps", type=int, default=120_000)
    parser.add_argument("--rolling_window", type=int, default=200)
    parser.add_argument("--restart_step", type=int, default=5_500)
    args = parser.parse_args()

    train, validation = load_records(resolve(args.log))
    steps = np.asarray([int(row["step"]) for row in train])
    losses = np.asarray([float(row["loss"]) for row in train])
    learning_rates = np.asarray([float(row["lr"]) for row in train])
    supervised = np.asarray([int(row.get("total_supervised_tokens", 0)) for row in train])
    seen = np.asarray([int(row.get("total_seen_tokens", 0)) for row in train])
    cumulative_seconds = np.asarray([float(row.get("cumulative_gpu_time_seconds", 0)) for row in train])
    smooth = rolling_mean(losses, args.rolling_window)

    latest = train[-1]
    latest_validation = validation[-1] if validation else None
    current_step = int(latest["step"])
    progress = min(1.0, current_step / args.max_steps)
    current_mean = statistics.fmean(losses[-500:])
    baseline_values = losses[(steps >= args.restart_step - 499) & (steps <= args.restart_step)]
    baseline_mean = float(baseline_values.mean()) if len(baseline_values) else float(losses[0])
    improvement = (baseline_mean - current_mean) / baseline_mean * 100

    restart_indices = np.flatnonzero(steps >= args.restart_step)
    restart_index = int(restart_indices[0]) if len(restart_indices) else 0
    elapsed = cumulative_seconds[-1] - cumulative_seconds[restart_index]
    completed = current_step - int(steps[restart_index])
    seconds_per_step = elapsed / completed if completed > 0 and elapsed > 0 else None
    remaining_hours = (args.max_steps - current_step) * seconds_per_step / 3600 if seconds_per_step else None

    interval = 200
    density_steps = []
    density_values = []
    for end in range(interval, len(train), interval):
        delta_seen = seen[end] - seen[end - interval]
        delta_supervised = supervised[end] - supervised[end - interval]
        if delta_seen > 0:
            density_steps.append(steps[end])
            density_values.append(100 * delta_supervised / delta_seen)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#75808c",
        "axes.labelcolor": "#4c5867",
        "xtick.color": "#596574",
        "ytick.color": "#596574",
    })
    figure = plt.figure(figsize=(16, 10), dpi=150, facecolor="#f4f5f1")
    grid = figure.add_gridspec(2, 1, height_ratios=[2.2, 1], left=0.075, right=0.95, top=0.72, bottom=0.09, hspace=0.36)
    loss_axis = figure.add_subplot(grid[0])
    schedule_axis = figure.add_subplot(grid[1])

    figure.text(0.075, 0.94, "TLGM 1B Reasoning SFT", fontsize=25, fontweight="bold", color="#19334a")
    figure.text(0.075, 0.906, f"Live progress through {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}", fontsize=11, color="#647080")
    metric_items = [
        ("STEP", f"{current_step:,} / {args.max_steps:,}"),
        ("PROGRESS", f"{progress * 100:.1f}%"),
        ("500-STEP LOSS", f"{current_mean:.4f}"),
        ("VALIDATION LOSS", f"{float(latest_validation['loss']):.4f}" if latest_validation else "n/a"),
        ("VALIDATION PPL", f"{float(latest_validation['perplexity']):.3f}" if latest_validation else "n/a"),
        ("ETA", f"{remaining_hours:.1f} hours" if remaining_hours is not None else "n/a"),
    ]
    for index, (label, value) in enumerate(metric_items):
        x = 0.075 + index * 0.147
        figure.text(x, 0.855, label, fontsize=8.5, fontweight="bold", color="#6c7784")
        figure.text(x, 0.82, value, fontsize=15, fontweight="bold", color="#19334a")
    progress_axis = figure.add_axes((0.075, 0.775, 0.875, 0.015))
    progress_axis.barh([0], [args.max_steps], color="#dde2df", height=1)
    progress_axis.barh([0], [current_step], color="#14857c", height=1)
    progress_axis.set_xlim(0, args.max_steps)
    progress_axis.axis("off")

    stride = max(1, len(steps) // 5000)
    loss_axis.plot(steps[::stride], losses[::stride], color="#9eb2b7", linewidth=0.65, alpha=0.25, label="Batch loss")
    loss_axis.plot(steps, smooth, color="#14857c", linewidth=2.2, label=f"Training loss ({args.rolling_window}-step mean)")
    if validation:
        val_steps = [int(row["step"]) for row in validation]
        val_losses = [float(row["loss"]) for row in validation]
        loss_axis.plot(val_steps, val_losses, color="#e57734", marker="o", markersize=4, linewidth=2.2, label="Validation loss")
    loss_axis.axvline(args.restart_step, color="#7b8490", linestyle="--", linewidth=1.2)
    loss_axis.annotate("Data/trainer fixes", (args.restart_step, loss_axis.get_ylim()[1]), xytext=(7, -18), textcoords="offset points", fontsize=9, color="#626c78")
    loss_axis.set_title("Cross-entropy loss", loc="left", fontsize=15, color="#253447")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_xlim(0, args.max_steps)
    loss_axis.xaxis.set_major_locator(MultipleLocator(20_000))
    loss_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    loss_axis.grid(True, color="#dce1e3", linewidth=0.7, alpha=0.9)
    loss_axis.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)

    schedule_axis.plot(steps, learning_rates * 1e6, color="#306998", linewidth=2, label="Learning rate")
    schedule_axis.set_title("Optimization schedule and target density", loc="left", fontsize=15, color="#253447")
    schedule_axis.set_ylabel("Learning rate (x 1e-6)")
    schedule_axis.set_xlabel("Optimizer step")
    schedule_axis.set_xlim(0, args.max_steps)
    schedule_axis.xaxis.set_major_locator(MultipleLocator(20_000))
    schedule_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    schedule_axis.grid(True, color="#dce1e3", linewidth=0.7, alpha=0.9)
    density_axis = schedule_axis.twinx()
    if density_steps:
        density_axis.plot(density_steps, density_values, color="#c59620", linewidth=1.4, alpha=0.85, label="Supervised-token density")
    density_axis.set_ylabel("Supervised targets (%)", color="#9b7416")
    density_axis.tick_params(axis="y", colors="#9b7416")
    lines = schedule_axis.lines + density_axis.lines
    schedule_axis.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right", fontsize=9)

    figure.text(
        0.075,
        0.025,
        f"Rolling loss improvement vs. steps {args.restart_step - 499:,}-{args.restart_step:,}: {improvement:.1f}%   |   "
        f"{len(train):,} train records   |   {len(validation):,} validation records",
        fontsize=10,
        color="#647080",
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "step": current_step,
        "max_steps": args.max_steps,
        "progress_percent": progress * 100,
        "latest_loss": float(latest["loss"]),
        "rolling_500_loss": current_mean,
        "pre_fix_500_loss": baseline_mean,
        "rolling_loss_improvement_percent": improvement,
        "validation_step": int(latest_validation["step"]) if latest_validation else None,
        "validation_loss": float(latest_validation["loss"]) if latest_validation else None,
        "validation_perplexity": float(latest_validation["perplexity"]) if latest_validation else None,
        "estimated_remaining_hours": remaining_hours,
        "output": str(output),
    }
    summary_path = resolve(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
