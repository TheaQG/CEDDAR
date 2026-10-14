from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.sigma_star_metrics import (
    load_sigma_star_metrics,
    plot_sigma_star_metric,
)


METRICS = ("r_lp", "slope_gen", "crps",)


def make_figure(eval_dir: str | Path, *, output: str | Path,):
    eval_dir = Path(eval_dir).expanduser()
    output = Path(output).expanduser()

    data = load_sigma_star_metrics(eval_dir)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 6.6),
        sharex="col",
    )

    # ---------------------------------------------------------
    # Top: across-date standard deviation
    # Bottom: nominal SEM
    # ---------------------------------------------------------

    for col, key in enumerate(METRICS):
        plot_sigma_star_metric(
            axes[0, col],
            data,
            key,
            error_mode="std",
            show_title=True,
            show_xlabel=False,
            show_legend=(key == "slope_gen"),
            annotate_context=(key == "crps"),
        )

        plot_sigma_star_metric(
            axes[1, col],
            data,
            key,
            error_mode="sem",
            show_title=False,
            show_xlabel=True,
            show_legend=False,
            annotate_context=False,
        )

        axes[0, col].tick_params(axis="x", labelbottom=False,)

    # ---------------------------------------------------------
    # Panel labels
    # ---------------------------------------------------------

    labels = ("a", "b", "c", "d", "e", "f")

    for ax, label in zip(axes.flat, labels,):
        ax.text(
            0.98,
            1.035,
            f"({label})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )

    # ---------------------------------------------------------
    # Row labels
    # ---------------------------------------------------------

    fig.text(
        0.005,
        0.72,
        "Across-date variability\n" + r"(mean $\pm$ SD)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.text(
        0.005,
        0.27,
        "Uncertainty in mean\n" + r"(mean $\pm$ SEM)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.94,
        bottom=0.11,
        wspace=0.31,
        hspace=0.27,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output.parent.mkdir(parents=True, exist_ok=True,)

    # Treat output as basename even if a suffix was supplied.
    if output.suffix:
        output = output.with_suffix("")

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")

    fig.savefig(
        str(png),
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        str(pdf),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Evaluation: {eval_dir}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    return fig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-dir", type=Path, required=True, help=("Sigma-star evaluation root containing tables/agg_summary.csv and tables/metrics_by_sigma.csv."),)
    parser.add_argument("--output", type=Path, required=True, help="Output basename without extension.",)

    args = parser.parse_args()

    make_figure(args.eval_dir,output=args.output,)


if __name__ == "__main__":
    main()