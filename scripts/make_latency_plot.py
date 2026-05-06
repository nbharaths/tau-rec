"""Generate the latency-vs-score Pareto plot for the paper.

Produces a single-column figure showing pass^1 vs median trial runtime
across all 7 models in Table 3.

pass^1 numbers come from Table 3 in the paper. Latency numbers come from
the trial_results.json archives where available; for runs not yet
archived locally (GPT-5.4 variants, Sonnet 4.6, Gemini 2.5 Flash) the
authors provided latency stats from their machine.

Output: results/paper_final/latency_plot.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

REPO = Path(__file__).resolve().parent.parent

# (full_label, short_label, pass^1, median trial runtime in seconds, family, marker)
POINTS = [
    ("DS V4 Flash (non-think)",   "DS-V4F",         0.546,  78.7, "deepseek", "o"),
    ("DS V4 Flash (think-high)",  "DS-V4F+T(hi)",   0.560, 101.1, "deepseek", "o"),
    ("DS V4 Flash (think-max)",   "DS-V4F+T(max)",  0.571, 103.4, "deepseek", "o"),
    ("GPT-5.4 (no thinking)",     "GPT-5.4",    0.471,  31.5, "openai",   "s"),
    ("GPT-5.4 (med thinking)",    "GPT-5.4+T",  0.551,  76.3, "openai",   "s"),
    ("GPT-5-mini",                "GPT-5-mini", 0.417, 151.4, "openai",   "s"),
    ("Sonnet 4.6",                "Sonnet 4.6", 0.537,  71.6, "anthropic","D"),
    ("Gemini 2.5 Flash",          "Gem-2.5F",   0.275,  33.4, "google",   "P"),
    ("Qwen3-32B",                 "Qwen3-32B",  0.271, 109.8, "qwen",     "^"),
]

FAMILY_COLORS = {
    "deepseek":  "#1f77b4",
    "openai":    "#d62728",
    "anthropic": "#9467bd",
    "google":    "#2ca02c",
    "qwen":      "#ff7f0e",
}

# Manual annotation offsets (dx, dy) in points (keyed by short_label) to avoid overlaps
ANNOT = {
    "DS-V4F":         (  6, -14),
    "DS-V4F+T(hi)":   ( 10,  10),
    "DS-V4F+T(max)":  ( 10, -10),
    "GPT-5.4":    (  8, -10),
    "GPT-5.4+T":  (-50,  10),
    "GPT-5-mini": (-50,  -10),
    "Sonnet 4.6": (-58,  -14),
    "Gem-2.5F":   ( 10,   0),
    "Qwen3-32B":  ( 10,   0),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    # DS V4 Flash thinking trajectory (off → high → max)
    ds = sorted([p for p in POINTS if "DS V4 Flash" in p[0]], key=lambda p: p[3])
    ax.plot(
        [p[3] for p in ds], [p[2] for p in ds],
        linestyle="--", color=FAMILY_COLORS["deepseek"], alpha=0.5, linewidth=1, zorder=1,
    )

    # GPT-5.4 thinking trajectory (no → med)
    gpt54 = sorted([p for p in POINTS if "GPT-5.4" in p[0] and "mini" not in p[0]], key=lambda p: p[3])
    ax.plot(
        [p[3] for p in gpt54], [p[2] for p in gpt54],
        linestyle="--", color=FAMILY_COLORS["openai"], alpha=0.5, linewidth=1, zorder=1,
    )

    for full_label, short_label, p1, t, family, marker in POINTS:
        ax.scatter(
            t, p1,
            color=FAMILY_COLORS[family],
            marker=marker, s=60, zorder=2,
            edgecolor="black", linewidth=0.6,
        )
        dx, dy = ANNOT.get(short_label, (5, 5))
        ax.annotate(
            short_label, (t, p1),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7,
        )

    # Approximate Pareto frontier line (visual cue)
    pareto = [
        ("GPT-5.4 (no thinking)", 31.5,  0.471),
        ("Sonnet 4.6",            71.6,  0.537),
        ("GPT-5.4 (med thinking)",76.3,  0.551),
        ("DS V4 Flash (think-max)",103.4, 0.571),
    ]
    pareto.sort(key=lambda p: p[1])
    ax.plot(
        [p[1] for p in pareto], [p[2] for p in pareto],
        linestyle="-", color="grey", alpha=0.4, linewidth=1.5, zorder=0,
        label="Pareto frontier",
    )

    ax.set_xlabel("Median trial runtime (s)", fontsize=8)
    # Match the .tex \passhat{1} macro which renders "pass^1" in upright text
    ax.set_ylabel("pass^1", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(25, 200)
    ax.set_ylim(0.22, 0.62)
    # Explicit, well-spaced x ticks
    ax.xaxis.set_major_locator(FixedLocator([30, 50, 80, 120, 200]))
    ax.xaxis.set_major_formatter(FixedFormatter(["30", "50", "80", "120", "200"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = REPO / "results/paper_final/latency_plot.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
