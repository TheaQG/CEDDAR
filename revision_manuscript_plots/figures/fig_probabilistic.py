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

    fig = plt.figure(figsize=(7.2, 5.9))

    outer = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 1.0),
        width_ratios=(1.18, 0.82),
        hspace=0.46,
        wspace=0.34,
    )

    # CRPS pair
    crps_grid = outer[0, 0].subgridspec(
        1,
        2,
        wspace=0.30,
    )

    ax_crps_all = fig.add_subplot(crps_grid[0, 0])
    ax_crps_wet = fig.add_subplot(crps_grid[0, 1])

    # Histogram
    ax_hist = fig.add_subplot(outer[0, 1])

    # Reliability triptych
    rel_grid = outer[1, 0].subgridspec(
        1,
        3,
        wspace=0.30,
    )

    ax_rel1 = fig.add_subplot(rel_grid[0, 0])

    ax_rel10 = fig.add_subplot(rel_grid[0, 1])

    ax_rel20 = fig.add_subplot(rel_grid[0, 2])

    # Spread-skill
    ax_spread = fig.add_subplot(outer[1, 1])

    probabilistic.crps(
        ax_crps_all,
        data,
        subset="all_land",
        label="(a)",
    )
    probabilistic.crps(
        ax_crps_wet,
        data,
        subset="observed_wet",
    )

    if args.histogram == "rank":
        probabilistic.ranks(
            ax_hist,
            data,
            label="(b)",
        )
    else:
        probabilistic.pit(
            ax_hist,
            legacy_prob,
            label="(b)",
        )

    probabilistic.reliability(
        ax_rel1,
        data,
        threshold=1.0,
        label="(c)",
    )
    probabilistic.reliability(
        ax_rel10,
        data,
        threshold=10.0,
    )
    probabilistic.reliability(
        ax_rel20,
        data,
        threshold=20.0,
    )
    probabilistic.spread_skill(
        ax_spread,
        data,
        source="revision",
        label="(d)",
    )

    # Compact triptych.
    ax_rel10.set_ylabel("")
    ax_rel20.set_ylabel("")

    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        top=0.97,
        bottom=0.11,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")

    style.save_figure(fig, output / "fig05_probabilistic",)

    plt.close(fig)

    print(output / "fig05_probabilistic.pdf")


if __name__ == "__main__":
    main()