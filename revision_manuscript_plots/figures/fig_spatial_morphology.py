"""Main manuscript figure:
PSD, SAL, and precipitation-object morphology.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import style
from ..data import legacy, revision
from ..panels import distributions, morphology
from ..paths import REVISION_ROOT


def default_baseline_eval(name):
    return (legacy.DEFAULT_EVALUATION.parent.parent / "baselines" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--legacy-eval", type=Path,)
    parser.add_argument("--qm-eval", type=Path,)
    parser.add_argument("--revision-dir", type=Path,)
    parser.add_argument("--sal-threshold", type=float, default=1.0,)
    parser.add_argument("--output-dir", type=Path,)

    args = parser.parse_args()

    style.apply_style()

    psd_data = legacy.load_psd(args.legacy_eval)
    qm_psd = legacy.load_psd(args.qm_eval or default_baseline_eval("qm"))
    morph = revision.load_morphology(args.revision_dir)

    fig = plt.figure(figsize=(7.2, 7.0))

    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(1.24, 0.78),
        hspace=0.40,
    )

    # ------------------------------------------------------------------
    # Upper row
    # ------------------------------------------------------------------

    top = outer[0].subgridspec(
        1,
        2,
        width_ratios=(2.25, 0.92),
        wspace=0.30,
    )

    ax_psd = fig.add_subplot(top[0, 0])

    sal_grid = top[0, 1].subgridspec(
        3,
        1,
        hspace=0.48,
    )

    ax_sal_s = fig.add_subplot(sal_grid[0, 0])
    ax_sal_a = fig.add_subplot(sal_grid[1, 0])
    ax_sal_l = fig.add_subplot(sal_grid[2, 0])

    distributions.psd(
        ax_psd,
        psd_data,
        baselines={
            "qm": qm_psd,
        },
        label="(a)",
    )

    ax_psd.legend(
        loc="upper right",
        fontsize=7.2,
        ncol=1,
    )

    morphology.sal(
        ax_sal_s,
        morph,
        component="S",
        threshold=args.sal_threshold,
        label="(b)",
    )
    morphology.sal(
        ax_sal_a,
        morph,
        component="A",
        threshold=args.sal_threshold,
    )
    morphology.sal(
        ax_sal_l,
        morph,
        component="L",
        threshold=args.sal_threshold,
    )

    # ------------------------------------------------------------------
    # Lower row: object morphology
    # ------------------------------------------------------------------

    object_grid = outer[1].subgridspec(
        1,
        4,
        wspace=0.42,
    )

    ax_abs_count = fig.add_subplot(object_grid[0, 0])
    ax_abs_fraction = fig.add_subplot(object_grid[0, 1])
    ax_equal_count = fig.add_subplot(object_grid[0, 2])
    ax_equal_fraction = fig.add_subplot(object_grid[0, 3])

    morphology.objects(
        ax_abs_count,
        morph,
        analysis="absolute",
        metric="object_count_absolute_error",
        label="(c)",
    )
    morphology.objects(
        ax_abs_fraction,
        morph,
        analysis="absolute",
        metric="largest_object_fraction",
    )
    morphology.objects(
        ax_equal_count,
        morph,
        analysis="equal_area",
        metric="object_count_absolute_error",
    )
    morphology.objects(
        ax_equal_fraction,
        morph,
        analysis="equal_area",
        metric="largest_object_fraction",
    )

    # Reduce visual repetition.
    for ax in (
        ax_abs_fraction,
        ax_equal_count,
        ax_equal_fraction,
    ):
        ax.set_ylabel("")

    order = [
        style.method_label(method)
        for method in (
            "era5_bilinear",
            "qm",
            "ceddar_members",
        )
    ]

    handles, labels = style.unique_legend(
        (
            ax_abs_count,
            ax_abs_fraction,
            ax_equal_count,
            ax_equal_fraction,
        ),
        order=order,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.995,
        top=0.98,
        bottom=0.12,
    )

    output = (args.output_dir or REVISION_ROOT / "manuscript_figures")
    style.save_figure(fig, output / "fig06_spatial_morphology",)

    plt.close(fig)

    print(output / "fig06_spatial_morphology.pdf")


if __name__ == "__main__":
    main()