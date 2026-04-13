"""
Read TensorBoard event files from out/tb_logs and save a single multiplot
figure to out/results/exp01_overview.png.
Run from the pretrain/ directory: python plot_exp01.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_LOG_DIR  = "../out/tb_logs"
RESULTS_DIR = "../out/results"

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
            print(f"  [skip] tag not found: {tag}")
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


def plot_overview(data: dict):
    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle("Exp 01 — Baseline Pretraining Overview", fontsize=14, fontweight="bold", y=1.01)

    # ── (0,0) Loss ────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    tr_s, tr_v = sv(data, "loss/train")
    va_s, va_v = sv(data, "loss/val")
    if tr_s:
        ax.plot(tr_s, tr_v, linewidth=0.7, color="steelblue", label="train")
    if va_s:
        ax.plot(va_s, va_v, linewidth=1.2, color="orange", marker="o",
                markersize=3, label="val")
    ax.set_title("Loss"); ax.set_ylabel("loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,1) Learning Rate ───────────────────────────────────────────────────
    ax = axes[0, 1]
    s, v = sv(data, "lr")
    ax.plot(s, v, linewidth=0.8, color="darkorange")
    ax.set_title("Learning Rate"); ax.set_ylabel("lr")
    ax.grid(True, alpha=0.3)

    # ── (1,0) Gradient Norm ───────────────────────────────────────────────────
    ax = axes[1, 0]
    gn_s, gn_v = sv(data, "grad_norm")
    gc_s, gc_v = sv(data, "grad_clipped")
    clipped_steps = [s for s, v in zip(gc_s, gc_v) if v > 0]
    if gn_s:
        ax.plot(gn_s, gn_v, linewidth=0.6, color="steelblue", label="grad_norm")
    if clipped_steps:
        norm_map = dict(zip(gn_s, gn_v))
        clipped_norms = [norm_map.get(s, 0) for s in clipped_steps]
        ax.scatter(clipped_steps, clipped_norms, color="red", s=8, zorder=3, label="clipped")
    ax.set_title("Gradient Norm"); ax.set_ylabel("norm")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,1) Parameter Norm ──────────────────────────────────────────────────
    ax = axes[1, 1]
    s, v = sv(data, "param_norm")
    ax.plot(s, v, linewidth=0.8, color="purple")
    ax.set_title("Parameter Norm"); ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.3)

    # ── (2,0) Update-to-Weight Ratio ──────────────────────────────────────────
    ax = axes[2, 0]
    s, v = sv(data, "update_to_weight_ratio")
    if s:
        ax.plot(s, v, linewidth=1.0, color="darkorange", marker="o", markersize=3)
    ax.axhline(1e-3, color="green", linestyle="--", linewidth=0.8, label="ideal ~1e-3")
    ax.axhline(1e-2, color="red",   linestyle="--", linewidth=0.8, label="high 1e-2")
    ax.axhline(1e-4, color="gray",  linestyle="--", linewidth=0.8, label="low 1e-4")
    ax.set_yscale("log")
    ax.set_title("Update-to-Weight Ratio"); ax.set_ylabel("‖Δw‖ / ‖w‖")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── (2,1) Activation Mean ─────────────────────────────────────────────────
    ax = axes[2, 1]
    ms, mv = sv(data, "activation/mean")
    if ms:
        ax.plot(ms, mv, linewidth=1.0, color="teal", marker="o", markersize=3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("ln_final Activation Mean"); ax.set_ylabel("mean")
    ax.grid(True, alpha=0.3)

    # ── (3,0) Activation Std ──────────────────────────────────────────────────
    ax = axes[3, 0]
    ss, sv_ = sv(data, "activation/std")
    if ss:
        ax.plot(ss, sv_, linewidth=1.0, color="coral", marker="o", markersize=3)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("ln_final Activation Std"); ax.set_ylabel("std")
    ax.grid(True, alpha=0.3)

    # ── (3,1) empty ───────────────────────────────────────────────────────────
    axes[3, 1].set_visible(False)

    # shared x-label on bottom visible axes
    for ax in axes[3, :1]:
        ax.set_xlabel("step")
    for ax in [axes[2, 0], axes[2, 1]]:
        ax.set_xlabel("")

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "exp01_overview.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


if __name__ == "__main__":
    print(f"Loading TB logs from {TB_LOG_DIR} ...")
    data = load_scalars(TB_LOG_DIR)
    print(f"Tags loaded: {list(data.keys())}")
    print("Generating overview plot ...")
    plot_overview(data)
    print("Done.")
