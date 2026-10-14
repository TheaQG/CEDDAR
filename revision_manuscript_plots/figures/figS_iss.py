from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.iss import (
    load_iss_data,
    plot_iss_threshold,
)


DEFAULT_THRESHOLDS = (
    1.0,
    5.0,
    10.0,
    20.0,
)


def make_figure(revision_dir: str | Path, *, output: str | Path,):
    data = load_iss_data(revision_dir)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 7.2),
        sharex=True,
        sharey=True,
    )

    for i, (ax, threshold,) in enumerate(zip(axes.flat, DEFAULT_THRESHOLDS,)):
        row = i // 2
        col = i % 2

        plot_iss_threshold(
            ax,
            data,
            threshold,
            show_xlabel=(row == 1),
            show_ylabel=(col == 0),
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

    handles, labels = (axes[0, 0].get_legend_handles_labels())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015,),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.95,
        bottom=0.15,
        wspace=0.20,
        hspace=0.22,
    )

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True,)

    if output.suffix:
        output = (output.with_suffix(""))
    
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")

    fig.savefig(str(png), dpi=300, bbox_inches="tight",)
    fig.savefig(str(pdf), bbox_inches="tight",)

    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision-dir", type=Path, required=True, help=("Directory containing the revision ISS tables."),)
    parser.add_argument("--output", type=Path, required=True, help=("Output file path without extension."),)

    args = parser.parse_args()

    make_figure(args.revision_dir, output=args.output,)


if __name__ == "__main__":
    main()