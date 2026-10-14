"""Main manuscript figure:
deterministic performance and event detection.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import style
from ..data import revision
from ..panels import deterministic
from ..paths import REVISION_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--revision-dir", type=Path,)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    style.apply_style()

    data = revision.load_deterministic(args.revision_dir)

    fig = plt.figure(figsize=(9.0, 6.0))

    outer = fig.add_gridspec(
        2,
        3,
        height_ratios=(0.92, 1.0,),
        hspace=0.48,
        wspace=0.22,
    )

    ax_mae = fig.add_subplot(outer[0, 0])
    ax_rmse = fig.add_subplot(outer[0, 1], sharey=ax_mae)
    ax_corr = fig.add_subplot(outer[0, 2], sharey=ax_mae)

    ax_pod = fig.add_subplot(outer[1, 0])
    ax_far = fig.add_subplot(outer[1, 1])
    ax_csi = fig.add_subplot(outer[1, 2])

    deterministic.daily_errors(
        ax_mae,
        data,
        metric="mae",
        label="(a)",
        show_method_labels=True,
        show_improvement=True,
    )
    deterministic.daily_errors(
        ax_rmse,
        data,
        metric="rmse",
        label="(b)",
        show_method_labels=False,
        show_improvement=True,
    )
    deterministic.spatial_correlation(
        ax_corr,
        data,
        label="(c)",
    )

    top_methods = ("era5_bilinear", "qm", "ceddar_mean", "ceddar_median", "ceddar_pmm")
    positions = range(1, len(top_methods) + 1)

    ax_mae.set_yticks(positions)
    ax_mae.set_yticklabels([style.method_label(method) for method in top_methods])
    ax_mae.set_xlim(-3, 4)

    ax_rmse.tick_params(axis="y", labelleft=False)
    ax_rmse.set_xlim(-4, 6)
    ax_corr.tick_params(axis="y", labelleft=False)

    deterministic.event_detection(
        ax_pod,
        data,
        metric="pod",
        label=None,#"(c)",
    )
    deterministic.event_detection(
        ax_far,
        data,
        metric="far",
        label=None,#"(d)",
    )
    deterministic.event_detection(
        ax_csi,
        data,
        metric="csi",
        label=None,#"(e)",
    )

    style.panel_label(ax_pod, "(d)", x=-0.17, y=1.08)
    style.panel_label(ax_far, "(e)", x=-0.17, y=1.08)
    style.panel_label(ax_csi, "(f)", x=-0.17, y=1.08)

    order = [
        style.method_label(method)
        for method in (
            "era5_bilinear",
            "qm",
            "ceddar_mean",
            "ceddar_median",
            "ceddar_pmm",
        )
    ]

    handles, labels = style.unique_legend(
        (
            ax_pod,
            ax_far,
            ax_csi,
        ),
        order=order,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.11,
        right=0.99,
        top=0.97,
        bottom=0.13,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")
    style.save_figure(fig, output / "fig03_deterministic",)

    plt.close(fig)

    print(output / "fig03_deterministic.pdf")


if __name__ == "__main__":
    main()