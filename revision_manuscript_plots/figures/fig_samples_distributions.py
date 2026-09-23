"""Main manuscript figure:
generated examples, seasonal distributions, and tail statistics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .. import style
from ..data import legacy
from ..panels import distributions
from ..paths import REVISION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--legacy-eval", type=Path,)
    parser.add_argument("--qm-eval", type=Path,)
    parser.add_argument("--generation-dir", type=Path,)
    parser.add_argument("--dates", nargs=2, default=("20190103", "20190104"),)
    parser.add_argument("--map-vmax", type=float, default=30.0,)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    style.apply_style()

    # ------------------------------------------------------------------
    # Saved legacy evaluation products
    # ------------------------------------------------------------------

    dist = legacy.load_seasonal_distributions(args.legacy_eval)
    tails = legacy.load_extremes(args.legacy_eval)

    qm_eval = (args.qm_eval or legacy.baseline_evaluation("qm"))
    qm_dist = legacy.load_seasonal_distributions(qm_eval)
    qm_tails = legacy.load_extremes(qm_eval)

    # ------------------------------------------------------------------
    # Physical example fields
    # ------------------------------------------------------------------

    examples = legacy.load_example_fields(args.dates, generation_dir=args.generation_dir,)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(7.2, 5.8))

    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.1, 1.15),
        height_ratios=(1.2, 1.0),
        hspace=0.38,
        wspace=0.24,
    )

    map_grid = outer[0, 0].subgridspec(
        2,
        6,
        wspace=0.06,
        hspace=0.08,
    )

    season_grid = outer[1, 0].subgridspec(
        1,
        4,
        wspace=0.18,
    )

    side_grid = outer[:, 1].subgridspec(
        2,
        1,
        hspace=0.42,
    )

    # ------------------------------------------------------------------
    # (a) Generated examples
    # ------------------------------------------------------------------

    norm = Normalize(vmin=0.0, vmax=args.map_vmax,)

    map_axes = []
    last_image = None

    columns = (
        ("era5_condition", None, "ERA5"),
        ("danra", None, "DANRA"),
        ("ceddar_members", 0, "CEDDAR member 1"),
        ("ceddar_members", 1, "CEDDAR member 2"),
        ("ceddar_members", 2, "CEDDAR member 3"),
        ("ceddar_pmm", None, "CEDDAR PMM"),
    )

    for row, date in enumerate(args.dates):
        for col, (
            method,
            member,
            title,
        ) in enumerate(columns):
            ax = fig.add_subplot(map_grid[row, col])
            map_axes.append(ax)

            last_image = distributions.example(
                ax,
                examples,
                date,
                method=method,
                member=member,
                norm=norm,
                origin="lower",
            )

            if row == 0:
                ax.set_title(title, pad=2,)

            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    (
                        f"{date[:4]}-"
                        f"{date[4:6]}-"
                        f"{date[6:]}"
                    ),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=7.5,
                    color=style.AXIS_GREY,
                )

            if row == 0 and col == 0:
                style.panel_label(
                    ax,
                    "(a)",
                    x=-0.28,
                    y=1.08,
                )

    cbar = fig.colorbar(
        last_image,
        ax=map_axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.055,
        aspect=45,
    )

    cbar.set_label(r"Precipitation (mm day$^{-1}$)")

    # ------------------------------------------------------------------
    # (b) Seasonal distributions
    # ------------------------------------------------------------------

    baseline_distributions = {"qm": qm_dist,}

    season_axes = []

    for index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
        ax = fig.add_subplot(season_grid[0, index])

        season_axes.append(ax)
        distributions.seasonal(
            ax,
            dist,
            season,
            baselines=baseline_distributions,
            label="(b)" if index == 0 else None,
        )

        if index > 0:
            ax.set_ylabel("")

    # ------------------------------------------------------------------
    # (c) Tail and wet-day statistics
    # ------------------------------------------------------------------

    baseline_tails = {"qm": qm_tails,}

    ax_tail = fig.add_subplot(side_grid[0, 0])

    distributions.tails(
        ax_tail,
        tails,
        metrics=(
            "P95",
            "P99",
            "P99.9",
            "P99.99",
        ),
        baselines=baseline_tails,
        label="(c)",
    )

    ax_tail.set_title("Upper-tail quantiles", loc="left",)
    ax_tail.tick_params(axis="x", rotation=35,)
    ax_wet = fig.add_subplot(side_grid[1, 0])

    distributions.tails(
        ax_wet,
        tails,
        metrics=(
            "wet_freq",
            "wet_hit_rate",
        ),
        baselines=baseline_tails,
    )

    ax_wet.set_title("Wet-day statistics", loc="left",)
    ax_wet.set_xticklabels(("Frequency", "Hit rate"), rotation=20, ha="right",)
    ax_wet.set_ylim(0, 1.05)

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    order = [
        style.method_label(method)
        for method in (
            "danra",
            "era5_bilinear",
            "qm",
            "ceddar_members",
            "ceddar_pmm",
        )
    ]

    handles, labels = style.unique_legend(
        (
            *season_axes,
            ax_tail,
            ax_wet,
        ),
        order=order,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.97,
        bottom=0.13,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")
    style.save_figure(fig, output / "fig02_samples_distributions", )

    plt.close(fig)

    print(output / "fig02_samples_distributions.pdf")


if __name__ == "__main__":
    main()