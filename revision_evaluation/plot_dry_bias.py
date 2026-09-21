"""Plot Group 3 dry-bias decomposition from saved CSV tables.

The figure compares DANRA, deterministic baselines, deterministic CEDDAR
ensemble summaries, and the distribution across individual CEDDAR ensemble
members.

Expected input tables
---------------------
conditional_intensity.csv
seasonal_decomposition.csv
ensemble_member_decomposition.csv
ensemble_member_conditional_intensity.csv
manifest.json
"""

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
    "ceddar_mean": ("CEDDAR mean", "#0072B2", "D"),
    "ceddar_median": ("CEDDAR median", "#009E73", "^"),
    "ceddar_pmm": ("CEDDAR PMM", "#AA4499", "v"),
}

MEMBER_STYLE = {
    "label": "CEDDAR members",
    "color": "#56B4E9",
    "marker": "o",
}

SEASONS = ("DJF", "MAM", "JJA", "SON")


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value else float("nan")


def member_id(row):
    """Read ensemble-member identifier robustly from CSV."""
    return int(row["member"])


def member_summary(values):
    """Return median and IQR across ensemble members."""
    values = np.asarray(values, dtype=float)

    return (
        np.nanmedian(values, axis=0),
        np.nanquantile(values, 0.25, axis=0),
        np.nanquantile(values, 0.75, axis=0),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input-dir", type=Path, required=True,)

    parser.add_argument("--output-dir",type=Path,)

    args = parser.parse_args()

    conditional_file = (args.input_dir / "conditional_intensity.csv")
    seasonal_file = (args.input_dir / "seasonal_decomposition.csv")
    member_decomposition_file = (args.input_dir / "ensemble_member_decomposition.csv")
    member_conditional_file = (args.input_dir / "ensemble_member_conditional_intensity.csv")
    manifest_file = (args.input_dir / "manifest.json")

    conditional = read_csv(conditional_file)
    seasonal = read_csv(seasonal_file)
    member_decomposition = read_csv(member_decomposition_file)
    member_conditional = read_csv(member_conditional_file)

    manifest = json.loads(manifest_file.read_text())

    output = prepare_output(
        args.input_dir,
        args.output_dir
        or args.input_dir.resolve().parent
        / "manuscript_ready/dry_bias",
        manifest,
        "dry_bias",
    )

    # ------------------------------------------------------------------
    # Deterministic lookup tables
    # ------------------------------------------------------------------

    cond = {
        row["method"]: row
        for row in conditional
    }

    seas = {
        (row["method"], row["season"]): row
        for row in seasonal
    }

    methods = [
        method
        for method in METHODS
        if method in cond
    ]

    # ------------------------------------------------------------------
    # Ensemble-member lookup tables
    # ------------------------------------------------------------------

    member_seas = {
        (
            member_id(row),
            row["season"],
        ): row
        for row in member_decomposition
    }

    member_cond = {
        member_id(row): row
        for row in member_conditional
    }

    members = sorted(
        set(member_cond)
        & {
            member
            for member, season
            in member_seas
            if season == "ALL"
        }
    )

    if not members:
        raise ValueError(
            "No common ensemble members found in "
            "ensemble_member_decomposition.csv and "
            "ensemble_member_conditional_intensity.csv"
        )

    print(
        f"Plotting {len(members)} ensemble members"
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.16,
        top=0.90,
        wspace=0.28,
        hspace=0.35,
    )

    # ==============================================================
    # (a) Overall wet-pixel frequency
    # ==============================================================

    ax = axes[0, 0]

    categories = [
        "danra",
        "era5_bilinear",
        "qm",
        "ceddar_members",
        "ceddar_mean",
        "ceddar_median",
        "ceddar_pmm",
    ]

    categories = [
        category
        for category in categories
        if (
            category == "ceddar_members"
            or category in methods
        )
    ]

    x = np.arange(
        len(categories)
    )

    for xpos, category in zip(
        x,
        categories,
    ):
        if category == "ceddar_members":

            values = np.array(
                [
                    number(
                        member_seas[(member, "ALL")],
                        "wet_frequency",
                    )
                    for member in members
                ],
                dtype=float,
            )

            median, q25, q75 = member_summary(
                values
            )

            # Fixed jitter so every run produces the same figure.
            jitter = np.linspace(
                -0.16,
                0.16,
                len(values),
            )

            ax.scatter(
                xpos + jitter,
                values,
                s=16,
                color=MEMBER_STYLE["color"],
                alpha=0.45,
                edgecolors="none",
                zorder=2,
            )

            # IQR across members.
            ax.vlines(
                xpos,
                q25,
                q75,
                color=MEMBER_STYLE["color"],
                linewidth=5,
                alpha=0.8,
                zorder=3,
            )

            # Median across members.
            ax.scatter(
                xpos,
                median,
                s=55,
                marker="D",
                color=MEMBER_STYLE["color"],
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )

        else:
            value = number(
                seas[(category, "ALL")],
                "wet_frequency",
            )

            ax.scatter(
                xpos,
                value,
                s=42,
                marker=METHODS[category][2],
                color=METHODS[category][1],
                zorder=3,
            )

    category_labels = []

    for category in categories:
        if category == "ceddar_members":
            category_labels.append(
                "CEDDAR\nmembers"
            )
        else:
            category_labels.append(
                METHODS[category][0]
            )

    ax.set_xticks(
        x,
        category_labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Wet pixel-day frequency"
    )

    ax.set_title(
        "(a) Precipitation occurrence",
        loc="left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ==============================================================
    # (b) Conditional wet-pixel quantiles
    # ==============================================================

    ax = axes[0, 1]

    probabilities = np.array(
        QUANTILES,
        dtype=float,
    )

    prob_names = [
        f"P{int(probability * 100)}"
        for probability in probabilities
    ]

    # Baselines/reference first.
    first_methods = [
        method
        for method in (
            "danra",
            "era5_bilinear",
            "qm",
        )
        if method in methods
    ]

    for method in first_methods:
        row = cond[method]

        ax.plot(
            probabilities,
            [
                number(
                    row,
                    f"p{int(probability * 100)}",
                )
                for probability in probabilities
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            label=METHODS[method][0],
            linewidth=1.4,
        )

    # Individual-member conditional quantiles:
    # one statistic per member at P50/P90/P99.
    member_quantiles = np.array(
        [
            [
                number(
                    member_cond[member],
                    f"p{int(probability * 100)}",
                )
                for probability in probabilities
            ]
            for member in members
        ],
        dtype=float,
    )

    member_median, member_q25, member_q75 = (
        member_summary(
            member_quantiles
        )
    )

    ax.fill_between(
        probabilities,
        member_q25,
        member_q75,
        color=MEMBER_STYLE["color"],
        alpha=0.20,
        linewidth=0,
    )

    ax.plot(
        probabilities,
        member_median,
        marker=MEMBER_STYLE["marker"],
        color=MEMBER_STYLE["color"],
        linewidth=2.0,
        label="CEDDAR members (median, IQR)",
        zorder=4,
    )

    # Deterministic CEDDAR summaries.
    summary_methods = [
        method
        for method in (
            "ceddar_mean",
            "ceddar_median",
            "ceddar_pmm",
        )
        if method in methods
    ]

    for method in summary_methods:
        row = cond[method]

        ax.plot(
            probabilities,
            [
                number(
                    row,
                    f"p{int(probability * 100)}",
                )
                for probability in probabilities
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            label=METHODS[method][0],
            linewidth=1.4,
        )

    ax.set_xticks(
        probabilities,
        prob_names,
    )

    ax.set_ylabel(
        r"Precipitation conditional on wet "
        r"(mm day$^{-1}$)"
    )

    ax.set_title(
        "(b) Wet-pixel intensity distribution",
        loc="left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ==============================================================
    # (c) Seasonal wet frequency
    # ==============================================================

    ax = axes[1, 0]

    sx = np.arange(
        len(SEASONS)
    )

    # Baselines/reference.
    for method in first_methods:
        ax.plot(
            sx,
            [
                number(
                    seas[(method, season)],
                    "wet_frequency",
                )
                for season in SEASONS
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            linewidth=1.4,
        )

    # Ensemble members.
    member_seasonal_frequency = np.array(
        [
            [
                number(
                    member_seas[(member, season)],
                    "wet_frequency",
                )
                for season in SEASONS
            ]
            for member in members
        ],
        dtype=float,
    )

    member_median, member_q25, member_q75 = (
        member_summary(
            member_seasonal_frequency
        )
    )

    ax.fill_between(
        sx,
        member_q25,
        member_q75,
        color=MEMBER_STYLE["color"],
        alpha=0.20,
        linewidth=0,
    )

    ax.plot(
        sx,
        member_median,
        marker=MEMBER_STYLE["marker"],
        color=MEMBER_STYLE["color"],
        linewidth=2.0,
        zorder=4,
    )

    # CEDDAR deterministic summaries.
    for method in summary_methods:
        ax.plot(
            sx,
            [
                number(
                    seas[(method, season)],
                    "wet_frequency",
                )
                for season in SEASONS
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            linewidth=1.4,
        )

    ax.set_xticks(
        sx,
        SEASONS,
    )

    ax.set_ylabel(
        "Wet pixel-day frequency"
    )

    ax.set_title(
        "(c) Seasonal occurrence",
        loc="left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ==============================================================
    # (d) Seasonal conditional wet mean
    # ==============================================================

    ax = axes[1, 1]

    # Baselines/reference.
    for method in first_methods:
        ax.plot(
            sx,
            [
                number(
                    seas[(method, season)],
                    "conditional_mean_wet",
                )
                for season in SEASONS
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            linewidth=1.4,
        )

    # Ensemble members.
    member_seasonal_intensity = np.array(
        [
            [
                number(
                    member_seas[(member, season)],
                    "conditional_mean_wet",
                )
                for season in SEASONS
            ]
            for member in members
        ],
        dtype=float,
    )

    member_median, member_q25, member_q75 = (
        member_summary(
            member_seasonal_intensity
        )
    )

    ax.fill_between(
        sx,
        member_q25,
        member_q75,
        color=MEMBER_STYLE["color"],
        alpha=0.20,
        linewidth=0,
    )

    ax.plot(
        sx,
        member_median,
        marker=MEMBER_STYLE["marker"],
        color=MEMBER_STYLE["color"],
        linewidth=2.0,
        zorder=4,
    )

    # CEDDAR deterministic summaries.
    for method in summary_methods:
        ax.plot(
            sx,
            [
                number(
                    seas[(method, season)],
                    "conditional_mean_wet",
                )
                for season in SEASONS
            ],
            marker=METHODS[method][2],
            color=METHODS[method][1],
            linewidth=1.4,
        )

    ax.set_xticks(
        sx,
        SEASONS,
    )

    ax.set_ylabel(
        r"Mean precipitation conditional on wet "
        r"(mm day$^{-1}$)"
    )

    ax.set_title(
        "(d) Seasonal wet-pixel intensity",
        loc="left",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    # ==============================================================
    # Shared legend and title
    # ==============================================================

    handles, labels = (
        axes[0, 1].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )

    wet_threshold = float(
        manifest["wet_threshold"]
    )

    fig.suptitle(
        rf"Dry-bias decomposition "
        rf"($P \geq {wet_threshold:.0f}$ "
        rf"mm day$^{{-1}}$ defines wet)"
    )

    save_figure(
        fig,
        output,
        "group3_dry_bias_decomposition",
    )

    print(
        f"Saved dry-bias figure in {output}"
    )


if __name__ == "__main__":
    main()