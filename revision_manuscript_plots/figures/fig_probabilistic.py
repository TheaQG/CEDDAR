"""Main manuscript figure:
probabilistic calibration and ensemble performance.

TODO:
    - Include nominal coverage (potentially trade CRPS seasonal for coverage)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import style
from ..data import legacy, revision
from ..panels import probabilistic
from ..paths import REVISION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--revision-dir", type=Path,)
    parser.add_argument("--legacy-eval", type=Path, help=("Needed only when --histogram pit is selected."),)
    parser.add_argument("--histogram", choices=("rank", "pit"), default="rank",)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    style.apply_style()

    data = revision.load_probabilistic(args.revision_dir)

    legacy_prob = None

    if args.histogram == "pit":
        legacy_prob = (legacy.load_probabilistic(args.legacy_eval))

    fig = plt.figure(figsize=(10.5, 6.0))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.2, 0.8),
        hspace=0.48,
    )

    # ------------------------------------------------------------------
    # Top row: (a) coverage, (b) rank/PIT, (c) spread-skill
    # ------------------------------------------------------------------

    top_grid = outer[0].subgridspec(
        1,
        3,
        wspace=0.34,
    )

    ax_coverage = fig.add_subplot(top_grid[0, 0])
    ax_hist = fig.add_subplot(top_grid[0, 1])
    ax_spread = fig.add_subplot(top_grid[0, 2])

    # ------------------------------------------------------------------
    # Bottom row: (d) centered reliability triptych
    # ------------------------------------------------------------------

    bottom_band = outer[1].subgridspec(
        1,
        3,
        width_ratios=(1,4,1),
        wspace=0.0,
    )

    rel_grid = bottom_band[0,1].subgridspec(
        1,
        3,
        wspace=0.18,
    )
    ax_rel1 = fig.add_subplot(rel_grid[0, 0])
    ax_rel10 = fig.add_subplot(rel_grid[0, 1], sharey=ax_rel1)
    ax_rel20 = fig.add_subplot(rel_grid[0, 2], sharey=ax_rel1)

    # Secondary axes for reliability-bin populations.
    count_rel1 = ax_rel1.twinx()
    count_rel10 = ax_rel10.twinx()
    count_rel20 = ax_rel20.twinx()

    # ------------------------------------------------------------------
    # Plot panels
    # ------------------------------------------------------------------

    probabilistic.coverage(ax_coverage, data, label="(a)",)
    if args.histogram == "rank":
        probabilistic.ranks(ax_hist, data, label="(b)",)
    else:
        probabilistic.pit(ax_hist, legacy_prob, label="(b)",)

    probabilistic.spread_skill(ax_spread, data, label="(c)",)

    probabilistic.reliability(ax_rel1, data, threshold=1.0, count_ax=count_rel1, label="(d)",)
    probabilistic.reliability(ax_rel10, data, threshold=10.0, count_ax=count_rel10,)
    probabilistic.reliability(ax_rel20, data, threshold=20.0, count_ax=count_rel20,)

    # ------------------------------------------------------------------
    # Reliability formatting
    # ------------------------------------------------------------------

    # Shared primary y-axis.
    ax_rel10.set_ylabel("")
    ax_rel20.set_ylabel("")

    ax_rel10.tick_params(axis="y", labelleft=False,)
    ax_rel20.tick_params(axis="y", labelleft=False,)

    # One shared x-label for all three reliability panels.
    for ax in (ax_rel1, ax_rel10, ax_rel20,):
        ax.set_xlabel("")

    # Use a common scale for all count bars so the single right-hand
    # count axis is meaningful across the triptych.
    count_axes = (count_rel1, count_rel10, count_rel20,)

    count_max = max(ax.get_ylim()[1] for ax in count_axes)

    for ax in count_axes:
        ax.set_ylim(0, count_max,)
        ax.tick_params(axis="y", labelsize=6.5,pad=2,)

    # Only display the count-axis tick labels on the rightmost panel.
    for ax in (count_rel1, count_rel10,):
        ax.tick_params(axis="y", right=False, labelright=False,)
        ax.set_ylabel("")

    count_rel20.set_ylabel("Cases\n(pixel-days)", fontsize=7,)

    # Unified x label for panel (d).
    fig.text(0.515, 0.055, "Forecast probability", ha="center", va="center",)

    fig.subplots_adjust(
        left=0.075,
        right=0.955,
        top=0.97,
        bottom=0.12,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")

    style.save_figure(fig, output / "fig05_probabilistic",)

    plt.close(fig)

    print(output / "fig05_probabilistic.pdf")


if __name__ == "__main__":
    main()