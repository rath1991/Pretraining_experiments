"""
Read TensorBoard event files from out/tb_logs and save all metric plots to out/results/.
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
    """
    Merges all event files in log_dir into a single dict of {tag: [(step, value), ...]}.
    Sorted by step so two-session runs stitch together correctly.
    """
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


def steps_values(data: dict, tag: str):
    pairs = data.get(tag, [])
    if not pairs:
        return [], []
    steps, values = zip(*pairs)
    return list(steps), list(values)


def save(fig, fname: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def plot_all(data: dict):

    # ── loss ──────────────────────────────────────────────────────────────────
    tr_s, tr_v = steps_values(data, "loss/train")
    va_s, va_v = steps_values(data, "loss/val")
    fig, ax = plt.subplots()
    if tr_s:
        ax.plot(tr_s, tr_v, label="train", linewidth=0.8)
    if va_s:
        ax.plot(va_s, va_v, label="val", linewidth=1.2, marker="o", markersize=3)
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Loss — Exp 01 Baseline")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "loss.png")

    # ── learning rate ─────────────────────────────────────────────────────────
    s, v = steps_values(data, "lr")
    fig, ax = plt.subplots()
    ax.plot(s, v, linewidth=0.8, color="orange")
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("Learning Rate — Exp 01 Baseline")
    ax.grid(True, alpha=0.3)
    save(fig, "lr.png")

    # ── gradient norm ─────────────────────────────────────────────────────────
    gn_s, gn_v = steps_values(data, "grad_norm")
    gc_s, gc_v = steps_values(data, "grad_clipped")
    clipped_steps = [s for s, v in zip(gc_s, gc_v) if v > 0]
    clipped_norms = []
    if clipped_steps and gn_s:
        norm_map = dict(zip(gn_s, gn_v))
        clipped_norms = [norm_map.get(s, 0) for s in clipped_steps]

    fig, ax = plt.subplots()
    if gn_s:
        ax.plot(gn_s, gn_v, linewidth=0.6, color="steelblue", label="grad_norm")
    if clipped_steps:
        ax.scatter(clipped_steps, clipped_norms, color="red", s=8, zorder=3, label="clipped")
    ax.set_xlabel("step"); ax.set_ylabel("grad norm"); ax.set_title("Gradient Norm — Exp 01 Baseline")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "grad_norm.png")

    # ── param norm ────────────────────────────────────────────────────────────
    s, v = steps_values(data, "param_norm")
    fig, ax = plt.subplots()
    ax.plot(s, v, linewidth=0.8, color="purple")
    ax.set_xlabel("step"); ax.set_ylabel("L2 norm"); ax.set_title("Parameter Norm — Exp 01 Baseline")
    ax.grid(True, alpha=0.3)
    save(fig, "param_norm.png")

    # ── update-to-weight ratio ────────────────────────────────────────────────
    s, v = steps_values(data, "update_to_weight_ratio")
    fig, ax = plt.subplots()
    if s:
        ax.plot(s, v, linewidth=1.0, color="darkorange", marker="o", markersize=3)
    ax.axhline(1e-3, color="green", linestyle="--", linewidth=0.8, label="ideal ~1e-3")
    ax.axhline(1e-2, color="red",   linestyle="--", linewidth=0.8, label="high 1e-2")
    ax.axhline(1e-4, color="gray",  linestyle="--", linewidth=0.8, label="low 1e-4")
    ax.set_yscale("log"); ax.set_xlabel("step")
    ax.set_ylabel("||dtheta|| / ||theta||"); ax.set_title("Update-to-Weight Ratio — Exp 01 Baseline")
    ax.legend(); ax.grid(True, alpha=0.3)
    save(fig, "update_ratio.png")

    # ── activations (ln_final) ────────────────────────────────────────────────
    ms, mv = steps_values(data, "activation/mean")
    ss, sv = steps_values(data, "activation/std")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    if ms:
        ax1.plot(ms, mv, linewidth=1.0, color="teal", marker="o", markersize=3)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("mean"); ax1.set_title("ln_final Activations — Exp 01 Baseline")
    ax1.grid(True, alpha=0.3)
    if ss:
        ax2.plot(ss, sv, linewidth=1.0, color="coral", marker="o", markersize=3)
    ax2.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("std"); ax2.set_xlabel("step"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, "activations.png")


if __name__ == "__main__":
    print(f"Loading TB logs from {TB_LOG_DIR} ...")
    data = load_scalars(TB_LOG_DIR)
    print(f"Tags loaded: {list(data.keys())}")
    print("Generating plots ...")
    plot_all(data)
    print("Done.")
