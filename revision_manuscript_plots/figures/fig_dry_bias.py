"""Main manuscript figure:
dry-bias occurrence/intensity decomposition.
"""

import argparse
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from .. import style
from ..data import revision, legacy
from ..data.legacy import _field, _npz
from ..panels import dry_bias
from ..paths import REVISION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--revision-dir", type=Path,)
    parser.add_argument("--spatial-dir", type=Path,)
    parser.add_argument("--generation-dir", type=Path, help="Final-model generation directory containing land-sea mask")
    parser.add_argument("--qm-spatial-dir", type=Path,)
    parser.add_argument("--derived-spatial-dir", type=Path,)
    parser.add_argument("--bias-type", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--output-dir", type=Path,)
    parser.add_argument("--spatial-mode", choices=("sum", "mean"), default="sum",)

    args = parser.parse_args()

    spatial_mode = args.spatial_mode
    if spatial_mode == "sum":
        spatial_field = "sum"
        colorbar_label = "Accumulated precipitation over available 2019 days (mm)"
    elif spatial_mode == "mean":
        spatial_field = "mean"
        colorbar_label = r"Mean daily precipitation (mm day$^{-1}$)"
    else:
        raise ValueError(f"Unsupported spatial mode: {spatial_mode}. Choose 'sum' or 'mean'.")

    style.apply_style()

    data = revision.load_dry_bias(args.revision_dir)
    spatial = legacy.load_dry_bias_spatial_maps(args.spatial_dir, qm_directory=args.qm_spatial_dir, derived_directory=args.derived_spatial_dir, field=spatial_field, year="2019")

    generation_dir = args.generation_dir
    land = legacy.load_land_mask(generation_dir=generation_dir)

    # Figure layout
    fig = plt.figure(figsize=(11.2, 9.5))

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.90, 1.30),
        hspace=0.30,
    )

    # (a) Spatial maps: six map columns + one narrow colorbar column
    map_grid = outer[0].subgridspec(
        2,
        7,
        width_ratios=(1, 1, 1, 1, 1, 1, 0.12),
        wspace=0.10,
        hspace=0.10,
    )

    diag_grid = outer[1].subgridspec(
        2,
        2,
        wspace=0.30,
        hspace=0.38,
    )

    map_methods = (
        "danra",
        "era5_bilinear",
        "qm",
        "ceddar_mean",
        "ceddar_median",
        "ceddar_pmm",
    )

    map_titles = {
        "danra": "DANRA",
        "era5_bilinear": "Bilinear\nERA5",
        "qm": "QM",
        "ceddar_mean": "CEDDAR\nMean",
        "ceddar_median": "CEDDAR\nMedian",
        "ceddar_pmm": "CEDDAR\nPMM",
    }

    absolute_axes = []
    bias_axes = []

    reference = np.asarray(spatial["danra"], dtype=float)

    # Shared scale for absolute precipitation maps
    absolute_values = []

    for method in map_methods:
        field = np.asarray(spatial[method], dtype=float)

        if land is not None:
            values = field[land >= 0.5]
        else:
            values = field.ravel()

        absolute_values.append(values[np.isfinite(values)])

    absolute_values = np.concatenate(absolute_values)
    abs_vmin = float(np.nanpercentile(absolute_values, 2))
    abs_vmax = float(np.nanpercentile(absolute_values, 98))

    # Relative differences from DANRA
    relative_fields = {}
    for method in map_methods[1:]:
        field = np.asarray(spatial[method], dtype=float)
        difference = np.full_like(reference, np.nan, dtype=float)

        finite = (np.isfinite(reference) & np.isfinite(field))
        
        if args.bias_type == "relative":
            valid = (finite & (reference > 1e-6))
            difference[valid] = 100 * (field[valid] - reference[valid]) / reference[valid]
        elif args.bias_type == "absolute":
            difference[finite] = field[finite] - reference[finite]

        relative_fields[method] = difference

    print("\nSpatial differences from DANRA")
    print("--------------------------------")

    for method, field in relative_fields.items():
        values = field[(land >= 0.5) & np.isfinite(field)]

        print(
            f"{method:16s}"
            f" min={np.min(values):8.1f}"
            f" p02={np.percentile(values, 2):8.1f}"
            f" mean={np.mean(values):8.1f}"
            f" p98={np.percentile(values, 98):8.1f}"
            f" max={np.max(values):8.1f}"
            f" negative={100*np.mean(values < 0):5.1f}%"
        )

    # Symmetric common bias scale
    bias_values = []

    for field in relative_fields.values():
        if land is not None:
            values = field[land >= 0.5]
        else:
            values = field.ravel()
        values = values[np.isfinite(values)]
        bias_values.append(np.abs(values))

    bias_values = np.concatenate(bias_values)

    bias_limit = float(np.nanpercentile(bias_values, 98))

    print(f"bias scale: ±{bias_limit:.1f}")

    # (a) Top row: precipitation
    absolute_image = None
    for col, method in enumerate(map_methods):
        ax = fig.add_subplot(map_grid[0, col])
        absolute_axes.append(ax)

        absolute_image = dry_bias.spatial_map(
            ax,
            spatial[method],
            land=land,
            relative=False,
            vmin=abs_vmin,
            vmax=abs_vmax,
        )

        field = np.asarray(spatial[method], dtype=float)
        valid_land = ((land >= 0.5) & np.isfinite(field))

        spatial_mean = float(np.nanmean(field[valid_land]))
        spatial_std = float(np.nanstd(field[valid_land]))

        if args.spatial_mode == "sum":
            stat_text = (rf"{spatial_mean:.0f} $\pm$ {spatial_std:.0f} mm")
        else:
            stat_text = (rf"{spatial_mean:.2f} $\pm$ {spatial_std:.2f} mm day$^{-1}$")

        ax.set_title(f"{map_titles[method]}\n{stat_text}", fontsize=7.8, pad=3, linespacing=1.15)

        if col == 0:
            style.panel_label(ax, "(a)", x=-0.28, y=1.10)

    # Shared absolute colorbar
    cax_abs = fig.add_subplot(map_grid[0, 6])
    cbar_abs = fig.colorbar(absolute_image, cax=cax_abs)
    # cbar_abs.set_label(colorbar_label, fontsize=8)
    abs_ticks = np.linspace(abs_vmin, abs_vmax, num=4)
    cbar_abs.set_ticks(abs_ticks) # type: ignore
    cbar_abs.set_ticklabels([f"{value:.0f}" for value in abs_ticks]) # type: ignore
    cbar_abs.ax.tick_params(labelsize=6.5, pad=2)


    # (a) Bottom row: relative difference from DANRA
    # First position is relative to itself, so leave blank.
    ax_blank = fig.add_subplot(map_grid[1, 0])
    ax_blank.axis("off")

    bias_image = None
    for col, method in enumerate(map_methods[1:], start=1):
        ax = fig.add_subplot(map_grid[1, col])
        bias_axes.append(ax)

        bias_image = dry_bias.spatial_map(
            ax,
            relative_fields[method],
            land=land,
            relative=True,
            vmin=-bias_limit,
            vmax=bias_limit,
        )


    # Shared relative-difference colorbar
    cax_bias = fig.add_subplot(map_grid[1, 6])
    cbar_bias = fig.colorbar(bias_image, cax=cax_bias, extend="both")
    # if args.bias_type == "relative":
    #     cbar_bias.set_label("Difference from DANRA (%)", fontsize=8)
    # elif args.bias_type == "absolute":
    #     cbar_bias.set_label("Absolute difference from DANRA", fontsize=8)
    bias_ticks = np.linspace(-bias_limit, bias_limit, num=5)
    cbar_bias.set_ticks(bias_ticks) # type: ignore
    if args.bias_type == "relative":
        cbar_bias.set_ticklabels([f"{value:.0f}" for value in bias_ticks]) # type: ignore
    else:
        cbar_bias.set_ticklabels([f"{value:.0f}" for value in bias_ticks]) # type: ignore
    cbar_bias.ax.tick_params(labelsize=6.5, pad=2)



    # Row labels
    absolute_axes[0].text(
        -0.18,
        0.5,
        ("Accumulated\nprecipitation (2019)" if args.spatial_mode == "sum" else "Mean daily\nprecipitation"),
        transform=absolute_axes[0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.5,
        color=style.AXIS_GREY,
    )

    bias_axes[0].text(
        -0.18,
        0.5,
        ("Difference\nfrom DANRA (%)" if args.bias_type == "relative" else "Absolute difference\nfrom DANRA"),
        transform=bias_axes[0].transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.5,
        color=style.AXIS_GREY,
    )


    # (b-e) Dry-bias diagnostics
    ax_occurrence = fig.add_subplot(diag_grid[0, 0])
    ax_intensity = fig.add_subplot(diag_grid[0, 1])
    ax_season_occurrence = fig.add_subplot(diag_grid[1, 0])
    ax_season_intensity = fig.add_subplot(diag_grid[1, 1])

    dry_bias.occurrence(ax_occurrence, data, label="(b)")
    dry_bias.conditional_intensity(ax_intensity, data, label="(c)")
    dry_bias.seasonal(ax_season_occurrence, data, metric="wet_frequency", label="(d)")
    dry_bias.seasonal(ax_season_intensity, data, metric="conditional_mean_wet", label="(e)")

    # Shared legend for diagnostics
    order = [style.method_label(method) for method in map_methods]

    handles, labels = style.unique_legend((ax_occurrence, ax_intensity, ax_season_occurrence, ax_season_intensity), order=order)

    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.012), frameon=False)


    # Final adjustments and layout
    fig.subplots_adjust(left=0.08, right=0.97, top=0.98, bottom=0.13)

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")

    style.save_figure(fig, output / "fig04_dry_bias")

    plt.close(fig)
    print(output / "fig04_dry_bias")
        

if __name__ == "__main__":
    main()