"""Render the marketing attack-containment bar chart for the README.

Output: figures/marketing_attack_containment.png

Numbers are from the paper Section 7.4 adversarial-corpus differential:
SKG (T) contains 13 of 13 attacks; the declared-capability baseline (E)
contains 5 of 13. Reproducible from `tests/test_containment_matrix.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    systems  = ["LLM-only\n(Baseline A)", "Declared-capability\n(Baseline E)", "SKG (T)"]
    contained = [0, 5, 13]
    total     = 13

    colors   = ["#d97070", "#e0b070", "#3aa86a"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.bar(systems, contained, color=colors, edgecolor="#314466", linewidth=0.6)
    ax.axhline(total, linestyle="--", linewidth=0.7, color="#314466")
    ax.text(2.45, total + 0.2, f"all {total} attacks", color="#314466", fontsize=9, ha="right")

    ax.set_ylabel("Adversarial attacks contained")
    ax.set_ylim(0, total + 1.5)
    ax.set_yticks(range(0, total + 1, 2))
    ax.set_title("Containment on the SKG adversarial corpus (n=13)")
    for bar, value in zip(bars, contained):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.2,
            f"{value} / {total}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#314466",
        )

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "marketing_attack_containment.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
