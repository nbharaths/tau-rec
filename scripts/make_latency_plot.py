"""Generate the latency-vs-score Pareto plot for the paper.

Produces a single-column figure showing pass^1 vs median trial runtime
across the configurations in Table 3.

Install plotting deps once:
    uv pip install matplotlib seaborn pandas

Output: results/paper_final/latency_plot.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import pandas as pd
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent

DATA = pd.DataFrame(
    [
        # full_label, short_label, p1, latency_s, family
        ("DS V4 Flash (non-think)",  "DS-V4F",        0.546,  78.7, "deepseek"),
        ("DS V4 Flash (think-high)", "DS-V4F+T(hi)",  0.560, 101.1, "deepseek"),
        ("DS V4 Flash (think-max)",  "DS-V4F+T(max)", 0.571, 103.4, "deepseek"),
        ("GPT-5.4 (no thinking)",    "GPT-5.4",       0.471,  31.5, "openai"),
        ("GPT-5.4 (med thinking)",   "GPT-5.4+T",     0.551,  76.3, "openai"),
        ("GPT-5-mini",               "GPT-5-mini",    0.417, 151.4, "openai"),
        ("Sonnet 4.6",               "Sonnet 4.6",    0.537,  71.6, "anthropic"),
        ("Gemini 2.5 Flash",         "Gem-2.5F",      0.275,  33.4, "google"),
        ("Qwen3-32B",                "Qwen3-32B",     0.271, 109.8, "qwen"),
    ],
    columns=["full_label", "short_label", "p1", "latency_s", "family"],
)

PALETTE = {
    "deepseek":  "#1f77b4",
    "openai":    "#d62728",
    "anthropic": "#9467bd",
    "google":    "#2ca02c",
    "qwen":      "#ff7f0e",
}

MARKERS = {
    "deepseek":  "o",
    "openai":    "s",
    "anthropic": "D",
    "google":    "P",
    "qwen":      "^",
}

# Annotation offsets in points (keyed by short_label) to avoid label overlap
ANNOT_OFFSETS = {
    "DS-V4F":        ( 6, -14),
    "DS-V4F+T(hi)":  (10,  10),
    "DS-V4F+T(max)": (10, -10),
    "GPT-5.4":       ( 8, -10),
    "GPT-5.4+T":     (-50, 10),
    "GPT-5-mini":    (-50, -10),
    "Sonnet 4.6":    (-58, -14),
    "Gem-2.5F":      (10,   0),
    "Qwen3-32B":     (10,   0),
}

# Approximate Pareto frontier (manually identified from DATA)
PARETO_LABELS = [
    "GPT-5.4",          # 31.5s, 0.471
    "Sonnet 4.6",       # 71.6s, 0.537
    "GPT-5.4+T",        # 76.3s, 0.551
    "DS-V4F+T(max)",    # 103.4s, 0.571
]


def main() -> None:
    sns.set_theme(style="white", context="paper", font_scale=0.9)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    # Pareto frontier line
    pareto = DATA[DATA["short_label"].isin(PARETO_LABELS)].sort_values("latency_s")
    sns.lineplot(
        data=pareto, x="latency_s", y="p1",
        color="grey", linestyle="-", alpha=0.4, linewidth=1.5,
        ax=ax, legend=False, zorder=0,
    )

    # Per-model scatter points (color + marker by family)
    sns.scatterplot(
        data=DATA, x="latency_s", y="p1",
        hue="family", style="family",
        palette=PALETTE, markers=MARKERS,
        s=60, edgecolor="black", linewidth=0.6,
        ax=ax, legend=False, zorder=2,
    )

    # Per-point text labels
    for _, row in DATA.iterrows():
        dx, dy = ANNOT_OFFSETS.get(row["short_label"], (5, 5))
        ax.annotate(
            row["short_label"],
            (row["latency_s"], row["p1"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=7,
        )

    ax.set_xlabel("Median trial runtime (s)", fontsize=8)
    # pass^1 in upright text
    ax.set_ylabel("pass^1", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(25, 200)
    ax.set_ylim(0.22, 0.62)
    ax.xaxis.set_major_locator(FixedLocator([30, 50, 80, 120, 200]))
    ax.xaxis.set_major_formatter(FixedFormatter(["30", "50", "80", "120", "200"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout()
    out = REPO / "results/paper_final/latency_plot.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
