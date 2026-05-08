"""Render the README hero image.

Single 1200x630 PNG. The shape is a flow from "Your task" through SKG
to one of two outcomes (cached procedure or LLM fallback), with the
runtime gate drawn as the load-bearing layer. Numbers and labels come
from `eval/results/h1_stats.json` and `tests/test_containment_matrix.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _box(ax, x, y, w, h, label, fill, fg="#1f2a44"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0,
        edgecolor="#1f2a44",
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=12,
        color=fg,
    )


def _arrow(ax, x0, y0, x1, y1, label=None):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.1,
        color="#1f2a44",
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2 + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1f2a44",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    fig.patch.set_facecolor("#f7f9fc")
    ax.set_facecolor("#f7f9fc")

    ax.text(
        6.0, 5.85,
        "Skill Knowledge Graph",
        ha="center", va="center",
        fontsize=22,
        color="#1f2a44",
        weight="bold",
    )
    ax.text(
        6.0, 5.40,
        "Capability-token enforcement for LLM-synthesized procedures.",
        ha="center", va="center",
        fontsize=12,
        color="#3a4a6a",
    )

    _box(ax, 0.5, 3.0, 2.0, 1.0, "Your task",            "#dde6f4")
    _box(ax, 3.2, 3.0, 2.6, 1.0, "SKG router\n(exact / FTS / vector / graph)", "#fff7da")
    _box(ax, 6.6, 4.1, 2.3, 0.95, "Hit: run verified node", "#dff5e0")
    _box(ax, 6.6, 1.95, 2.3, 0.95, "Miss: fall through to LLM", "#fde2e2")

    _arrow(ax, 2.5, 3.5, 3.2, 3.5)
    _arrow(ax, 5.8, 3.7, 6.6, 4.55, label="80% (small)\n47.5% (large)")
    _arrow(ax, 5.8, 3.3, 6.6, 2.42, label="otherwise")

    _box(ax, 9.5, 3.0, 2.0, 1.0, "Wasmtime\nruntime gate", "#1f2a44", fg="#f7f9fc")
    _arrow(ax, 8.9, 4.55, 9.5, 3.85)
    _arrow(ax, 8.9, 2.42, 9.5, 3.15)

    ax.text(
        6.0, 0.95,
        "Adversarial corpus n=13: SKG contains 13. "
        "Declared-capability baseline contains 5. "
        "Source: tests/test_containment_matrix.py",
        ha="center", va="center",
        fontsize=11,
        color="#1f2a44",
    )
    ax.text(
        6.0, 0.50,
        "Latency on hits: 0.16 ms p50 vs 3082 ms p50 for the LLM call.",
        ha="center", va="center",
        fontsize=11,
        color="#1f2a44",
    )

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "hero.png"
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
