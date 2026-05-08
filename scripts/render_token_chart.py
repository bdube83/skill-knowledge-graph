"""Render the marketing token-savings bar chart for the README.

Output: figures/marketing_token_savings.png

Numbers are from `eval/results/baseline_a_large_report.json` and the
SKG router run captured in `eval/results/h1_stats.json` (this turn).
On the larger-context corpus (200 tasks, median ~1265 input tokens),
Baseline A consumes 1458.12 input tokens per task; SKG consumes 929.77,
saving 528 tokens (~36%).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    systems  = ["LLM-only\n(Baseline A)", "SKG (T)"]
    tokens   = [1458.12, 929.77]
    colors   = ["#d97070", "#3aa86a"]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.bar(systems, tokens, color=colors, edgecolor="#314466", linewidth=0.6)
    ax.set_ylabel("Input tokens per task (mean)")
    ax.set_title("Input-token cost on the larger-context corpus (n=200)")
    ax.set_ylim(0, max(tokens) * 1.18)

    for bar, value in zip(bars, tokens):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 30,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#314466",
        )

    saving = tokens[0] - tokens[1]
    pct    = 100 * saving / tokens[0]
    ax.annotate(
        f"saves {saving:.0f} tokens / task\n({pct:.1f}% reduction)",
        xy=(1, tokens[1]),
        xytext=(1.0, tokens[0] - 50),
        ha="center",
        fontsize=10,
        color="#314466",
        arrowprops=dict(arrowstyle="->", color="#314466", lw=0.7),
    )

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "figures" / "marketing_token_savings.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
