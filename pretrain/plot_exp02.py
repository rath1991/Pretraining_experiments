"""
Read TensorBoard event files from out/exp02_varX/tb_logs/ for all four variants
and save a single multiplot comparison figure to out/results/exp02_overview.png.
Run from the pretrain/ directory: python plot_exp02.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS_DIR = "../out/results"

VARIANTS = {
    "A — Adam / no warmup":    "../out/exp02_varA/tb_logs",
    "B — Adam / warmup 50":    "../out/exp02_varB/tb_logs",
    "C — AdamW / no warmup":   "../out/exp02_varC/tb_logs",
    "D — AdamW / warmup 50":   "../out/exp02_varD/tb_logs",
}

COLORS = {
    "A — Adam / no warmup":    "#e74c3c",   # red
    "B — Adam / warmup 50":    "#e67e22",   # orange
    "C — AdamW / no warmup":   "#2980b9",   # blue
    "D — AdamW / warmup 50":   "#27ae60",   # green
}

TAGS = [
    "loss/train",
    "loss/val",
    "lr",
    "grad_norm",
    "grad_clipped",
    "param_norm",
    "update_to_weight_ratio",
    "activation/mean",
    "activation/std",
]


def load_scalars(log_dir: str) -> dict:
    """Merge all event files in log_dir into {tag: [(step, value), ...]} sorted by step."""
    acc = EventAccumulator(log_dir)
    acc.Reload()
    available = acc.Tags()["scalars"]
    data = {}
    for tag in TAGS:
        if tag not in available:
            continue
        events = acc.Scalars(tag)
        data[tag] = sorted((e.step, e.value) for e in events)
    return data


def sv(data: dict, tag: str):
    """Return (steps_list, values_list) for a tag, or ([], []) if absent."""
    pairs = data.get(tag, [])
    if not pairs:
        return [], []
    steps, values = zip(*pairs)
    return list(steps), list(values)


def plot_overview(all_data: dict):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        "Exp 02 — Optimizer + Warmup Stability\n"
        "A: Adam/no-warmup   B: Adam/warmup   C: AdamW/no-warmup   D: AdamW/warmup (baseline)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    def plot_tag(ax, tag, title, ylabel, yscale="linear", hlines=None):
        for label, data in all_data.items():
            s, v = sv(data, tag)
            if s:
                ax.plot(s, v, linewidth=0.9, color=COLORS[label], label=label)
        if hlines:
            for y, color, ls, lbl in hlines:
                ax.axhline(y, color=color, linestyle=ls, linewidth=0.8, label=lbl)
        ax.set_title(title); ax.set_ylabel(ylabel)
        if yscale != "linear":
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)

    # ── (0,0) Train Loss ──────────────────────────────────────────────────────
    plot_tag(axes[0, 0], "loss/train", "Train Loss", "loss")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    # ── (0,1) Val Loss ────────────────────────────────────────────────────────
    for label, data in all_data.items():
        s, v = sv(data, "loss/val")
        if s:
            axes[0, 1].plot(s, v, linewidth=1.1, color=COLORS[label],
                            marker="o", markersize=3, label=label)
    axes[0, 1].set_title("Val Loss"); axes[0, 1].set_ylabel("loss")
    axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)

    # ── (1,0) Learning Rate ───────────────────────────────────────────────────
    plot_tag(axes[1, 0], "lr", "Learning Rate", "lr")

    # ── (1,1) Gradient Norm ───────────────────────────────────────────────────
    ax = axes[1, 1]
    for label, data in all_data.items():
        gn_s, gn_v = sv(data, "grad_norm")
        if gn_s:
            ax.plot(gn_s, gn_v, linewidth=0.7, color=COLORS[label], label=label)
    ax.set_title("Gradient Norm"); ax.set_ylabel("norm")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (2,0) Parameter Norm ──────────────────────────────────────────────────
    plot_tag(axes[2, 0], "param_norm", "Parameter Norm", "L2 norm")

    # ── (2,1) Update-to-Weight Ratio ──────────────────────────────────────────
    ax = axes[2, 1]
    for label, data in all_data.items():
        s, v = sv(data, "update_to_weight_ratio")
        if s:
            ax.plot(s, v, linewidth=1.0, color=COLORS[label],
                    marker="o", markersize=3, label=label)
    ax.axhline(1e-3, color="black", linestyle="--", linewidth=0.8, label="ideal ~1e-3")
    ax.axhline(1e-2, color="gray",  linestyle=":",  linewidth=0.8, label="high 1e-2")
    ax.axhline(1e-4, color="gray",  linestyle="-.", linewidth=0.8, label="low 1e-4")
    ax.set_yscale("log")
    ax.set_title("Update-to-Weight Ratio"); ax.set_ylabel("‖Δw‖ / ‖w‖")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # ── (3,0) Activation Mean ─────────────────────────────────────────────────
    ax = axes[3, 0]
    for label, data in all_data.items():
        s, v = sv(data, "activation/mean")
        if s:
            ax.plot(s, v, linewidth=1.0, color=COLORS[label],
                    marker="o", markersize=3, label=label)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("ln_final Activation Mean"); ax.set_ylabel("mean")
    ax.set_xlabel("step")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (3,1) Activation Std ──────────────────────────────────────────────────
    ax = axes[3, 1]
    for label, data in all_data.items():
        s, v = sv(data, "activation/std")
        if s:
            ax.plot(s, v, linewidth=1.0, color=COLORS[label],
                    marker="o", markersize=3, label=label)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("ln_final Activation Std"); ax.set_ylabel("std")
    ax.set_xlabel("step")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "exp02_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


if __name__ == "__main__":
    all_data = {}
    for label, log_dir in VARIANTS.items():
        print(f"Loading {label} from {log_dir} ...")
        all_data[label] = load_scalars(log_dir)
        print(f"  tags: {list(all_data[label].keys())}")
    print("Generating overview plot ...")
    plot_overview(all_data)
    print("Done.")
