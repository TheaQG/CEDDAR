from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.temporal_persistence import (
    load_temporal_persistence,
    plot_autocorrelation,
    plot_spell_distribution,
)


def make_figure(
    temporal_dir: str | Path,
    *,
    year: int = 2019,
    wet_threshold_mm: float = 1.0,
    output: str | Path,
):
    data = load_temporal_persistence(
        temporal_dir,
        year=year,
        wet_threshold_mm=(wet_threshold_mm),
        require_complete_year=True,
    )

    fig = plt.figure(figsize=(11.5, 7.0))

    gs = fig.add_gridspec(
        2,
        4,
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.13,
        height_ratios=[1.0, 1.15],
        hspace=0.42,
        wspace=0.28,
    )

    ax_ac = fig.add_subplot(gs[0, 1:3])
    ax_wet = fig.add_subplot(gs[1, 0:2])
    ax_dry = fig.add_subplot(gs[1, 2:4], sharey=ax_wet)

    plot_autocorrelation(
        ax_ac,
        data,
    )

    plot_spell_distribution(
        ax_wet,
        data,
        kind="wet",
        show_ylabel=True,
    )

    plot_spell_distribution(
        ax_dry,
        data,
        kind="dry",
        show_ylabel=False,
    )

    # Shared y-axis. 
    ax_dry.tick_params(axis="y", labelleft=False)

    for i, ax in enumerate([ax_ac, ax_wet, ax_dry]):
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

    # ---------------------------------------------------------
    # One common legend.
    # ---------------------------------------------------------

    handles, labels = (ax_ac.get_legend_handles_labels())

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015,),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True,exist_ok=True,)

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
    parser = argparse.ArgumentParser(description=("Generate the supplementary temporal-persistence figure from the saved legacy temporal diagnostics."))

    parser.add_argument("--temporal-dir", type=Path, required=True, help=("Temporal evaluation directory (.../temporal or .../temporal/tables)."),)
    parser.add_argument("--year", type=int, default=2019, help=("Evaluation year. The legacy diagnostics are accepted only if the saved dates form a complete daily calendar year."),)
    parser.add_argument("--wet-threshold-mm", type=float, default=1.0, help=("Threshold used by the saved temporal evaluation to classify the daily land-domain mean as wet/dry."),)
    parser.add_argument("--output", type=Path, required=True, help=("Output path without extension."),)

    args = parser.parse_args()

    make_figure(
        args.temporal_dir,
        year=args.year,
        wet_threshold_mm=(args.wet_threshold_mm),
        output=args.output,
    )


if __name__ == "__main__":
    main()
