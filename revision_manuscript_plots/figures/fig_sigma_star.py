from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ..panels.sigma_star import (
    load_sigma_star_data,
    load_selected_sample,
    list_selected_samples,
    get_example_fields,
    plot_psd,
    plot_metric,
    plot_example_strip,
)


DEFAULT_EXAMPLE_SIGMAS = [0.80, 0.90, 1.00, 1.10, 1.20]


def _resolve_model(root: Path, model: str | None):
    eval_root = root / "evaluation"
    gen_root = root / "samples" / "generation"

    if model is None:
        candidates = sorted(
            p.parent.parent.name
            for p in eval_root.glob("*/prcp/sigma_control")
        )

        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely infer model name. "
                f"Found: {candidates}. Pass --model explicitly."
            )

        model = candidates[0]

    eval_dir = eval_root / model / "prcp" / "sigma_control"
    gen_dir = gen_root / model

    if not eval_dir.exists():
        raise FileNotFoundError(eval_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(gen_dir)

    return model, eval_dir, gen_dir


def make_figure(
    eval_dir: str | Path,
    *,
    example_category: str = "heavy_mean",
    example_date: str | None = None,
    member_index: int | None = None,
    example_source: str = "member",
    example_sigmas=DEFAULT_EXAMPLE_SIGMAS,
    mask_ocean: bool = False,
    output: str | Path = "fig_sigma_star",
):
    eval_dir = Path(eval_dir).expanduser()
    output = Path(output).expanduser()

    data = load_sigma_star_data(eval_dir)

    example = load_selected_sample(eval_dir, category=example_category, date=example_date,)

    fields = get_example_fields(
        example,
        sigma_values=list(example_sigmas),
        source=example_source,
        member_index=member_index,
        mask_ocean=False,
    )
    example_mask = example.arrays.get("mask")

    fig = plt.figure(figsize=(15.2, 9.0))

    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.55, 1.25],
        height_ratios=[2.5, 1.05],
        hspace=0.30,
        wspace=0.28,   # more room between PSD colourbar and b-d
    )

    # (a)
    ax_a = fig.add_subplot(outer[0, 0])
    plot_psd(ax_a, data)

    # (b)--(d)
    right = outer[0, 1].subgridspec(
        3,
        1,
        hspace=0.32,
    )

    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0], sharex=ax_b)
    ax_d = fig.add_subplot(right[2, 0], sharex=ax_b)

    plot_metric(ax_b, data, "r_lp", error_mode="sem")
    plot_metric(ax_c, data, "slope_gen", error_mode="sem")
    plot_metric(ax_d, data, "crps", error_mode="sem")

    # Only bottom panel carries x tick labels / xlabel
    ax_b.set_xlabel("")
    ax_c.set_xlabel("")
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_c.tick_params(axis="x", labelbottom=False)

    # (e)
    n_examples = len(example_sigmas)

    bottom = outer[1, :].subgridspec(
        1,
        n_examples + 1,
        width_ratios=[1] * n_examples + [0.045],
        wspace=0.07,
    )

    example_axes = [
        fig.add_subplot(bottom[0, i])
        for i in range(n_examples)
    ]

    cax_e = fig.add_subplot(bottom[0, -1])

    im = plot_example_strip(
        example_axes,
        fields,
        mask=example_mask,
        mask_ocean=mask_ocean,
        show_outline=True,
        outline_alpha=0.45,
        outline_lw=0.7,
        percentile=99.5,
    )

    cb = fig.colorbar(
        im,
        cax=cax_e,
        orientation="vertical",
        # fraction=0.015,
        # pad=0.012,
    )
    cb.set_label(r"Precipitation (mm day$^{-1}$)", fontsize=11,)
    cb.ax.tick_params(labelsize=9.5)

    # Panel labels
    # Panel (a)
    ax_a.text(-0.10, 1.04, "(a)", transform=ax_a.transAxes, fontsize=14, fontweight="bold", va="bottom",)

    # Panels (b)--(d): upper-right
    for ax, label in ((ax_b, "b"), (ax_c, "c"), (ax_d, "d"),):
        ax.text(1.02, 1.04, f"({label})", transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="bottom", clip_on=False,)

    example_axes[0].text(-0.16, 1.04, "(e)", transform=example_axes[0].transAxes, fontsize=14, fontweight="bold", va="bottom",)

    # Reproducibility information without cluttering every panel.
    example_axes[0].set_ylabel(
        f"{example.date}\n{example.category.replace('_', ' ')}",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.96,
        top=0.97,
        bottom=0.07,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")

    fig.savefig(str(png), dpi=300, bbox_inches="tight",)
    fig.savefig(str(pdf), bbox_inches="tight",)

    print(f"Evaluation: {eval_dir}")
    print(f"Example: {example.category}, {example.date}")
    print(f"Bundle: {example.path}")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    return fig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-dir", type=Path, required=True, help="sigma_control evaluation directory",)
    parser.add_argument("--example-category", default="heavy_mean",)
    parser.add_argument("--example-date", default=None,)
    parser.add_argument("--member-index", type=int, default=None,help=("Original ensemble-member index. Defaults to the first saved paired member."),)
    parser.add_argument("--example-source", choices=["member", "pmm", "ens_mean"], default="member",)
    parser.add_argument("--example-sigmas", nargs="+", type=float, default=DEFAULT_EXAMPLE_SIGMAS,)
    parser.add_argument("--output", type=Path, default=Path("fig_sigma_star"),)
    parser.add_argument("--list-examples", action="store_true",)
    parser.add_argument("--spatial-mode", choices=["outline", "land", "both"], default="outline", help="How panel (e) is displayed",)

    args = parser.parse_args()

    if args.list_examples:
        print(list_selected_samples(args.eval_dir).to_string(index=False))
        return

    if args.spatial_mode == "both":

        make_figure(
            args.eval_dir,
            example_category=args.example_category,
            example_date=args.example_date,
            member_index=args.member_index,
            example_source=args.example_source,
            example_sigmas=args.example_sigmas,
            mask_ocean=False,
            output=Path(str(args.output) + "_outline"),
        )
        make_figure(
                    args.eval_dir,
                    example_category=args.example_category,
                    example_date=args.example_date,
                    member_index=args.member_index,
                    example_source=args.example_source,
                    example_sigmas=args.example_sigmas,
                    mask_ocean=True,
                    output=Path(str(args.output) + "_land"),
                )
    else:
        if args.spatial_mode == "outline":
            make_figure(
                args.eval_dir,
                example_category=args.example_category,
                example_date=args.example_date,
                member_index=args.member_index,
                example_source=args.example_source,
                example_sigmas=args.example_sigmas,
                mask_ocean=False,
                output=Path(str(args.output) + "_outline"),
            )
        elif args.spatial_mode == "land":
            make_figure(
                args.eval_dir,
                example_category=args.example_category,
                example_date=args.example_date,
                member_index=args.member_index,
                example_source=args.example_source,
                example_sigmas=args.example_sigmas,
                mask_ocean=True,
                output=Path(str(args.output) + "_land"),
            )


if __name__ == "__main__":
    main()