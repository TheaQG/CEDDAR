from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.daily_mean_bias import (
    load_daily_mean_bias,
    plot_daily_mean_bias,
)


def make_figure(
    revision_dir: str | Path,
    *,
    rolling_window: int = 14,
    output: str | Path,
):
    data = load_daily_mean_bias(revision_dir)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10.5, 4.4),
    )

    plot_daily_mean_bias(
        ax,
        data,
        rolling_window=(rolling_window),
    )

    handles, labels = (ax.get_legend_handles_labels())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, -0.1,),
        ncol=3,
        frameon=True,
        fontsize=9.5,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.80,
        bottom=0.16,
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
    parser = argparse.ArgumentParser(description=("Generate the supplementary daily domain-mean precipitation-bias figure."))

    parser.add_argument("--revision-dir", type=Path, required=True, help=("Directory containing daily_mean_bias.csv and daily_mean_bias_summary.csv."),)
    parser.add_argument("--rolling-window", type=int, default=14, help=("Trailing rolling-mean window in evaluation days."),)
    parser.add_argument("--output", type=Path, required=True, help=("Output path without extension."),)

    args = parser.parse_args()

    make_figure(args.revision_dir, rolling_window=(args.rolling_window), output=args.output,)


if __name__ == "__main__":
    main()
