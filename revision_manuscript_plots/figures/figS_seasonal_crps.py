from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.seasonal_crps import (
    load_seasonal_crps,
    plot_seasonal_crps,
)


def make_figure(revision_dir: str | Path, *, output: str | Path,):
    data = load_seasonal_crps(revision_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.4),
        sharex=True,
        sharey=False,
    )

    subsets = ("all_land", "observed_wet",)

    for i, (ax, subset,) in enumerate(zip(axes, subsets,)):
        plot_seasonal_crps(
            ax,
            data,
            subset=subset,
            show_ylabel=(i == 0),
        )

        ax.text(
            0.98,
            1.035,
            f"({chr(ord('a') + i)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )

    # The panels intentionally use independent y-ranges:
    # observed-wet CRPS has a substantially broader upper tail.
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.90,
        bottom=0.19,
        wspace=0.22,
    )

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True,)

    if output.suffix:
        output = (output.with_suffix(""))

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")

    fig.savefig(str(png),dpi=300, bbox_inches="tight",)
    fig.savefig(str(pdf), bbox_inches="tight",)

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    parser = argparse.ArgumentParser(description=("Generate the supplementary seasonal CRPS figure from revised probabilistic-evaluation products."))

    parser.add_argument("--revision-dir", type=Path, required=True, help=("Directory containing crps.csv and crps_summary.csv."),)
    parser.add_argument("--output", type=Path, required=True, help=("Output path without extension."),)

    args = parser.parse_args()

    make_figure(args.revision_dir, output=args.output,)


if __name__ == "__main__":
    main()
