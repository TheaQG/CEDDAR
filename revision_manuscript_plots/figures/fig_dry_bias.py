"""Main manuscript figure:
dry-bias occurrence/intensity decomposition.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import style
from ..data import revision
from ..panels import dry_bias
from ..paths import REVISION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--revision-dir", type=Path,)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    style.apply_style()

    data = revision.load_dry_bias(args.revision_dir)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
    )

    (
        ax_occurrence,
        ax_intensity,
        ax_season_occurrence,
        ax_season_intensity,
    ) = axes.ravel()

    dry_bias.occurrence(
        ax_occurrence,
        data,
        label="(a)",
    )
    dry_bias.conditional_intensity(
        ax_intensity,
        data,
        label="(b)",
    )
    dry_bias.seasonal(
        ax_season_occurrence,
        data,
        metric="wet_frequency",
        label="(c)",
    )
    dry_bias.seasonal(
        ax_season_intensity,
        data,
        metric="conditional_mean_wet",
        label="(d)",
    )

    order = [
        style.method_label(method)
        for method in (
            "danra",
            "era5_bilinear",
            "qm",
            "ceddar_members",
            "ceddar_mean",
            "ceddar_median",
            "ceddar_pmm",
        )
    ]

    handles, labels = style.unique_legend(
        axes.ravel(),
        order=order,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.97,
        bottom=0.18,
        wspace=0.30,
        hspace=0.38,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")
    style.save_figure(fig, output / "fig04_dry_bias",)

    plt.close(fig)

    print(output / "fig04_dry_bias.pdf")


if __name__ == "__main__":
    main()