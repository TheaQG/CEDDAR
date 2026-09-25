"""Main manuscript figure:
generated examples, seasonal distributions, and tail statistics.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .. import style
from ..data import legacy
from ..panels import distributions
from ..paths import REVISION_ROOT

def pooled_limits(examples, date, columns,):
    """Common physical precipitation scale for one example date."""

    day = examples["dates"][date]
    arrays = []

    for method, member, _ in columns:
        if member is None:
            field = day["fields"][method]
        else:
            field = day["ensemble"][member]

        field = np.asarray(field, dtype=float,)
        finite = field[np.isfinite(field)]

        if finite.size:
            arrays.append(finite)

    if not arrays:
        return 0.0, 1.0

    values = np.concatenate(arrays)

    # Precipitation is non-negative in these physical artifacts.
    return 0.0, float(np.nanmax(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--legacy-eval", type=Path,)
    parser.add_argument("--qm-eval", type=Path,)
    parser.add_argument("--generation-dir", type=Path,)
    parser.add_argument("--dates", nargs=2, default=("20190103", "20190104"),)
    parser.add_argument("--ensemble-histograms", type=Path, help="Path to previously derived ensemble histograms.",)
    parser.add_argument("--output-dir", type=Path,)
    parser.add_argument("--tails-mode", choices=("lines", "bars"), default="lines", help="Plot seasonal tail quantiles as connected lines or grouped bars.")

    args = parser.parse_args()

    style.apply_style()

    # ------------------------------------------------------------------
    # Saved legacy evaluation products
    # ------------------------------------------------------------------

    dist = legacy.load_seasonal_distributions(args.legacy_eval)
    ensemble_dist = (legacy.load_ensemble_histograms(args.ensemble_histograms) if args.ensemble_histograms else None)

    qm_eval = (args.qm_eval or legacy.baseline_evaluation("qm"))
    qm_dist = legacy.load_seasonal_distributions(qm_eval)

    # ------------------------------------------------------------------
    # Physical example fields
    # ------------------------------------------------------------------

    examples = legacy.load_example_fields(args.dates, generation_dir=args.generation_dir,)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(12, 7.2))

    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(1.18, 1.0, 0.62),
        hspace=0.4,
    )

    map_grid = outer[0].subgridspec(
        2,
        6,
        wspace=0.1,
        hspace=0.05,
    )

    season_grid = outer[1].subgridspec(
        1,
        4,
        wspace=0.16,
    )

    tail_grid = outer[2].subgridspec(
        1,
        4,
        wspace=0.13,
    )

    # ------------------------------------------------------------------
    # (a) Generated examples
    # ------------------------------------------------------------------

    map_axes = []
    last_image = None

    columns = (
        ("era5_condition", None, "ERA5"),
        ("danra", None, "DANRA"),
        ("ceddar_members", 0, "CEDDAR\nmember 1"),
        ("ceddar_members", 1, "CEDDAR\nmember 2"),
        ("ceddar_members", 2, "CEDDAR\nmember 3"),
        ("ceddar_pmm", None, "CEDDAR\nPMM"),
    )

    row_limits = {date: pooled_limits(examples, date, columns,) for date in args.dates}

    for row, date in enumerate(args.dates):
        vmin, vmax = (row_limits[date])

        for col, (method, member, title,) in enumerate(columns):
            ax = fig.add_subplot(map_grid[row, col])

            distributions.example(
                ax,
                examples,
                date,
                method=method,
                member=member,
                vmin=vmin,
                vmax=vmax,
                variable="prcp",
                show_ocean=True,
                add_outline=True,
                add_boxplot=True,
                add_colorbar=(col == len(columns) - 1), # only add colorbar on last field
            )

            if row == 0:
                ax.set_title(title, fontsize=7.6, pad=3)
            if col == 0:
                ax.text(-0.10, 0.5, (f"{date[:4]}-{date[4:6]}-{date[6:]}"), transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7.2, color=style.AXIS_GREY,)
            if row == 0 and col == 0:
                style.panel_label(ax, "(a)", x=-0.34, y=1.10,
                )


    # ------------------------------------------------------------------
    # (b) Seasonal distributions
    # ------------------------------------------------------------------

    baseline_distributions = {"qm": qm_dist,}

    season_axes = []
    season_results = {}

    for index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
        if season_axes:
            ax = fig.add_subplot(season_grid[0, index], sharey=season_axes[0],)
        else:
            ax = fig.add_subplot(season_grid[0, index])

        season_axes.append(ax)

        season_results[season] = (
            distributions.seasonal(
                ax,
                dist,
                season,
                baselines=baseline_distributions,
                ensemble=ensemble_dist,
                label=("(b)" if index == 0 else None),
                xlim=(-5,120),
                show_percentiles=True,
            )
        )
        ax.set_xlabel("")

        if index > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    ymin = min(ax.get_ylim()[0] for ax in season_axes)
    ymax = max(ax.get_ylim()[1] for ax in season_axes)
    for ax in season_axes:
        ax.set_ylim(ymin, ymax)
        ax.tick_params(labelbottom=False)

    fig.supxlabel(r"Precipitation (mm day$^{-1}$)", x=0.50, y=0.105, fontsize=9)
    # fig.text(0.50, 0.335, r"Precipitation (mm day$^{-1}$)", ha="center", va="center", fontsize=9)
    # ------------------------------------------------------------------
    # (c) Tail and wet-day statistics
    # ------------------------------------------------------------------

    baseline_tails = {"qm": qm_dist,}
    tail_axes = []

    for index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
        ax = fig.add_subplot(tail_grid[0, index], sharex=season_axes[index],)

        tail_axes.append(ax)

        distributions.seasonal_tails(
            ax,
            dist,
            season,
            baselines=baseline_tails,
            ensemble=ensemble_dist,
            mode=args.tails_mode,
            label=("(c)" if index == 0 else None),
        )

        ax.set_xlim(-5, 120)

        if index > 0:
            ax.set_yticklabels([])


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
            *tail_axes,
        ),
        order=order,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.025),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.98,
        bottom=0.15,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")
    style.save_figure(fig, output / "fig02_samples_distributions", )

    plt.close(fig)

    print(output / "fig02_samples_distributions.pdf")


if __name__ == "__main__":
    main()