"""Plot Group 3 dry-bias decomposition from saved CSV tables"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for plotting

import matplotlib.pyplot as plt
import numpy as np

from .plot_common import prepare_output, save_figure
from .dry_bias_decomposition import QUANTILES 

METHODS = {
    "danra": ("DANRA", "black", "o"),
    "era5_bilinear": ("Bilinear ERA5", "#555555", "o"),
    "qm": ("QM", "#C48100", "s"),
    "cedddar_mean": ("CEDDAR mean", "#0072B2", "D"),
    "ceddar_median": ("CEDDAR median", "#009E73", "^"),
    "ceddar_pmm": ("CEDDAR PMM", "#AA4499", "v"),
}

SEASONS = ("DJF", "MAM", "JJA", "SON")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, required=True,)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    conditional_file = (args.input_dir / "conditional_intensity.csv")
    seasonal_file = (args.input_dir / "seasonal_decomposition.csv")
    manifest_file = (args.input_dir / "manifest.json")

    conditional = read_csv(conditional_file)
    seasonal = read_csv(seasonal_file)  
    manifest = json.loads(manifest_file.read_text())

    output = prepare_output(
        args.input_dir,
        args.output_dir or args.input_dir.resolve().parent / "manuscript_ready/dry_bias",
        manifest,
        "dry_bias",
    )

    cond = {row["method"]: row for row in conditional}
    seas = {row["method"]: row for row in seasonal}
    methods = [m for m in METHODS if m in cond]

    x = np.arange(len(methods))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.15,
        top=0.90,
        wspace=0.28,
        hspace=0.35,
    )

    # a) Overall wet-pixel frequency
    ax = axes[0, 0]

    values = [number(seas[(m, "ALL")], "wet_frequency",) for m in methods]

    ax.scatter(x, values, s=42, color=[METHODS[m][1] for m in methods], zorder=3)
    ax.set_xticks(x, [METHODS[m][0] for m in methods], rotation=35, ha="right",)
    ax.set_ylabel("Wet pixel-day frequency")
    ax.set_title("(a) Precipitation occurrence", loc="left")
    ax.grid(axis="y", alpha=0.25)


    # b) Conditional wet-pixel quantiles
    ax = axes[0, 1]

    probabilities = np.array(QUANTILES)
    prob_names = [f"P{int(p*100)}%" for p in probabilities]

    for m in methods:
        row = cond[m]

        ax.plot(probabilities,
                [number(row, f"wet_intensity_q{int(p*100)}") for p in probabilities],
                marker=METHODS[m][2],
                color=METHODS[m][1],
                label=METHODS[m][0],
                linewidth=1.4,
        )

    ax.set_xticks(probabilities, prob_names)
    ax.set_ylabel(r"Precipitation conditional on wet (mm day$^{-1}$)")
    ax.set_title("(b) Wet-pixel intensity distribution", loc="left")
    ax.grid(axis="y", alpha=0.25)


    # c) Seasonal wet frequency
    ax = axes[1, 0]

    sx = np.arange(len(SEASONS))

    for m in methods:
        ax.plot(sx, [number(seas[(m, s)], "wet_frequency",) for s in SEASONS],
                marker=METHODS[m][2],
                color=METHODS[m][1],
                linewidth=1.4,
        )

    ax.set_xticks(sx, SEASONS)
    ax.set_ylabel("Wet pixel-day frequency")
    ax.set_title("(c) Seasonal occurrence", loc="left")
    ax.grid(axis="y", alpha=0.25)


    # d) Seasonal conditional mean
    ax = axes[1, 1]

    for m in methods:
        ax.plot(sx, [number(seas[(m, s)], "conditional_mean_wet",) for s in SEASONS],
                marker=METHODS[m][2],
                color=METHODS[m][1],
                linewidth=1.4,
        )

    ax.set_xticks(sx, SEASONS)
    ax.set_ylabel(r"Mean precipitation conditional on wet (mm day$^{-1}$)")
    ax.set_title("(d) Seasonal wet-pixel intensity", loc="left")
    ax.grid(axis="y", alpha=0.25)

    handles, labels = (axes[0, 1].get_legend_handles_labels())

    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(rf"Dry-bias decomposition "
                 rf"($P \geq {manifest['wet_threshold']:.0f}$"
                 rf"mm day$^{{-1}}$ defines wet)"
                 )

    save_figure(fig, output, "group3_dry_bias_decomposition")

    print(f"Saved dry-bias figure in {output}")


if __name__ == "__main__":
    main()