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

    fig = plt.figure(figsize=(7.2, 6.0))

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.92, 1.0),
        hspace=0.48,
    )
    top = outer[0].subgridspec(
        1,
        2,
        wspace=0.42,
    )
    bottom = outer[1].subgridspec(
        1,
        3,
        wspace=0.35,
    )

    ax_mae = fig.add_subplot(top[0, 0])
    ax_rmse = fig.add_subplot(top[0, 1])

    ax_pod = fig.add_subplot(bottom[0, 0])
    ax_far = fig.add_subplot(bottom[0, 1])
    ax_csi = fig.add_subplot(bottom[0, 2])

    deterministic.daily_errors(
        ax_mae,
        data,
        metric="mae",
        label="(a)",
    )
    deterministic.daily_errors(
        ax_rmse,
        data,
        metric="rmse",
        label="(b)",
    )
    deterministic.event_detection(
        ax_pod,
        data,
        metric="pod",
        label="(c)",
    )
    deterministic.event_detection(
        ax_far,
        data,
        metric="far",
        label="(d)",
    )
    deterministic.event_detection(
        ax_csi,
        data,
        metric="csi",
        label="(e)",
    )

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