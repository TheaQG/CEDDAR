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
    """
    Common precipitation scale for one example date,
    calculated from Danish-land pixels only.
    """
    day = examples["dates"][date]

    land = np.asarray(day["land"])
    land = np.squeeze(land) >= 0.5

    arrays = []

    for method, member, _ in columns:
        if member is None:
            field = day["fields"][method]
        else:
            field = day["ensemble"][member]

        field = np.asarray(field, dtype=float,)
        field = np.squeeze(field)

        if field.shape != land.shape:
            raise ValueError(f"{date}: field shape {field.shape} does not match land mask {land.shape}")

        values = field[land & np.isfinite(field)]

        if values.size:
            arrays.append(values)

    if not arrays:
        return 0.0, 1.0

    values = np.concatenate(arrays)

    return (0.0, float(np.nanmax(values)),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--legacy-eval", type=Path,)
    parser.add_argument("--qm-histograms", type=Path, required=True, help=("Corrected land-only QM daily histogram product"))
    parser.add_argument("--generation-dir", type=Path,)
    parser.add_argument("--dates", nargs=2, default=("20191011", "20200926"),) #default=("20190103", "20190104"),) 
    # ("20190810", "20191011")
    # ("20200209", "20190209")
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

    qm_histograms = (legacy.load_qm_histograms(args.qm_histograms))

    # ------------------------------------------------------------------
    # Physical example fields
    # ------------------------------------------------------------------

    examples = legacy.load_example_fields(args.dates, generation_dir=args.generation_dir,)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(12, 7.7))

    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=(
            1.40,
            1.0,
            0.62,
        ),
        hspace=0.36,
    )

    map_grid = outer[0].subgridspec(
        2,
        6,
        wspace=0.02,
        hspace=0.12,
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

            is_lr = (method == "era5_condition")

            distributions.example(
                ax,
                examples,
                date,
                method=method,
                member=member,
                vmin=vmin,
                vmax=vmax,
                variable="prcp",
                ocean_style=("tint" if is_lr else "hatch"),
                ocean_alpha=0.38,
                add_outline=True,
                add_boxplot=True,
                add_colorbar=(col == len(columns) - 1),
            )

            if row == 0:
                ax.set_title(title, fontsize=7.6, pad=3)
            if col == 0:
                ax.text(-0.10, 0.5, (f"{date[:4]}-{date[4:6]}-{date[6:]}"), transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9, color=style.AXIS_GREY,)
            if row == 0 and col == 0:
                style.panel_label(ax, "(a)", x=-0.34, y=1.10,)


    # ------------------------------------------------------------------
    # (b) Seasonal distributions
    # ------------------------------------------------------------------

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
                ensemble=ensemble_dist,
                qm_histograms=qm_histograms,
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

    tail_axes = []

    for index, season in enumerate(("DJF", "MAM", "JJA", "SON")):
        ax = fig.add_subplot(tail_grid[0, index], sharex=season_axes[index],)

        tail_axes.append(ax)

        distributions.seasonal_tails(
            ax,
            dist,
            season,
            ensemble=ensemble_dist,
            qm_histograms=qm_histograms,
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