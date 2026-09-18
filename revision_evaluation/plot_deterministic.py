"""Plot saved Group 1 metrics; no model loading, inference or metric changes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .plot_common import prepare_output, write_plot_metadata


# The same colours and markers identify methods in both figures.
METHODS = {
    "era5_bilinear": ("Bilinear ERA5", "#555555", "o"),
    "qm": ("QM", "#C48100", "s"),
    "ceddar_mean": ("CEDDAR mean", "#0072B2", "D"),
    "ceddar_median": ("CEDDAR median", "#009E73", "^"),
    "ceddar_pmm": ("CEDDAR PMM", "#AA4499", "v"),
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    files = [args.input_dir / name for name in
             ("daily_continuous_metrics.csv", "event_detection_metrics.csv", "manifest.json")]
    daily, events = (read_csv(path) for path in files[:2])
    manifest = json.loads(files[2].read_text())
    dates = sorted(manifest["dates"])
    lookup = {(row["date"], row["method"]): row for row in daily}
    expected = {(date, method) for date in dates for method in METHODS}
    if len(dates) != len(set(dates)) or len(lookup) != len(daily) or set(lookup) != expected:
        raise ValueError("Daily metrics must contain exactly one row per common date and method.")
    if any(row["subset"] != "all_land" for row in daily):
        raise ValueError("Expected all-land daily metrics.")
    support = [int(lookup[date, "era5_bilinear"]["n_pixels"]) for date in dates]
    if any(int(lookup[date, method]["n_pixels"]) != support[i]
           for i, date in enumerate(dates) for method in METHODS):
        raise ValueError("Pixel support differs between methods.")
    if sum(support) != manifest["n_pixel_days"] or min(support) <= 0:
        raise ValueError("Pixel support does not match the manifest.")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.axisbelow": True})
    args.output_dir = prepare_output(args.input_dir,
        args.output_dir or args.input_dir.resolve().parent / "manuscript_ready/deterministic",
        manifest, "deterministic")
    summary = {}
    comparison = list(METHODS)[1:]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.26, top=0.78, wspace=0.65)
    for panel, metric, letter in zip(axes, ("mae", "rmse"), ("a", "b")):
        values, labels = [], []
        for method in comparison:
            delta = np.array([float(lookup[date, method][metric]) -
                              float(lookup[date, "era5_bilinear"][metric]) for date in dates])
            if not np.isfinite(delta).all():
                raise ValueError(f"Non-finite paired {metric} for {method}.")
            fraction = float(np.mean(delta < 0))  # Exact ties do not count as improvements.
            summary[f"{method}_{metric}"] = {
                "n_dates": len(delta), "fraction_lower_error": fraction,
                "n_ties": int(np.sum(delta == 0)), "mean_difference": float(delta.mean()),
                "median_difference": float(np.median(delta)),
                "minimum": float(delta.min()), "maximum": float(delta.max()),
            }
            values.append(delta)
            labels.append(f"{METHODS[method][0]}\n{100*fraction:.1f}% lower error")
        boxes = panel.boxplot(values, vert=False, widths=0.48, patch_artist=True,
                              whis=1.5, showfliers=True, showmeans=True,
                              medianprops={"color": "black", "linewidth": 1.4},
                              meanprops={"marker": "D", "markersize": 4,
                                         "markerfacecolor": "black", "markeredgecolor": "white"},
                              flierprops={"marker": "o", "markersize": 2.4,
                                          "markerfacecolor": "none", "alpha": 0.65})
        for i, method in enumerate(comparison):
            colour = METHODS[method][1]
            boxes["boxes"][i].set(facecolor=colour, edgecolor=colour, alpha=0.55)
            boxes["fliers"][i].set(markeredgecolor=colour)
        panel.axvline(0, color="#444444", linewidth=1, linestyle="--", zorder=0)
        panel.set_yticks(range(1, 5), labels)
        panel.invert_yaxis()
        panel.tick_params(axis="y", length=0, labelsize=9)
        panel.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        panel.set_title(f"({letter}) Daily {metric.upper()} difference", loc="left", pad=12)
        panel.set_xlabel(r"Method $-$ bilinear ERA5 (mm day$^{-1}$)")
        panel.margins(x=0.08)
    fig.suptitle("Paired daily errors relative to bilinear ERA5", y=0.97, fontsize=13)
    fig.text(0.5, 0.885, f"{len(dates)} common test dates · negative differences favour the compared method",
             ha="center", fontsize=10)
    fig.text(0.5, 0.095, "Boxes: interquartile range; line: median; diamond: mean.\n"
             "Whiskers: 1.5 × IQR; all outliers shown. Percentages: dates with strictly lower error.",
             ha="center", va="center", fontsize=9, linespacing=1.5)
    save_figure(fig, args.output_dir, "group1_daily_error_differences")

    thresholds = [1.0, 10.0, 20.0]
    event_lookup = {(row["method"], float(row["threshold"])): row for row in events}
    if len(event_lookup) != len(events) or set(event_lookup) != {
            (method, threshold) for method in METHODS for threshold in thresholds}:
        raise ValueError("Expected one pooled detection row per method and threshold.")
    for row in events:
        if (row["subset"] != "all_land" or row["season"] != "ALL"
                or int(row["n_pixel_days"]) != sum(support) or int(row["n_dates"]) != len(dates)):
            raise ValueError("Detection support differs from daily metrics.")
        h, m, f, c = (int(row[key]) for key in ("hits", "misses", "false_alarms", "correct_negatives"))
        if h+m+f+c != sum(support):
            raise ValueError("Contingency counts do not match pixel support.")
        for key, calculated in (("pod", h/(h+m)), ("far", f/(h+f)), ("csi", h/(h+m+f))):
            if not np.isclose(float(row[key]), calculated, rtol=0, atol=1e-12):
                raise ValueError(f"Incorrect pooled {key} in input table.")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.30, top=0.77, wspace=0.25)
    for panel, key, title, direction, letter in zip(
            axes, ("pod", "far", "csi"),
            ("Probability of detection", "False-alarm ratio", "Critical success index"),
            ("Higher is better", "Lower is better", "Higher is better"), ("a", "b", "c")):
        for method, (label, colour, marker) in METHODS.items():
            panel.plot(thresholds, [float(event_lookup[method, t][key]) for t in thresholds],
                       color=colour, marker=marker, label=label, markersize=5,
                       linewidth=1.4, markeredgecolor="white", markeredgewidth=0.5)
        panel.set(xlim=(-0.5, 21.5), ylim=(0, 1), xticks=thresholds,
                  yticks=np.linspace(0, 1, 6), xlabel=r"Threshold (mm day$^{-1}$)")
        panel.set_title(f"({letter}) {title}\n{direction}", loc="left", pad=10)
        panel.grid(color="#E5E5E5", linewidth=0.6)
    fig.suptitle("Deterministic event detection", y=0.98, fontsize=13)
    fig.text(0.5, 0.90, f"{len(dates)} test dates · {sum(support):,} common land pixel-days · event ≥ threshold",
             ha="center", fontsize=10)
    handles = [Line2D([], [], color=c, marker=m, label=l, linewidth=1.4, markersize=5)
               for l, c, m in METHODS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncol=5, frameon=False, fontsize=9, columnspacing=1.4, handlelength=1.6)
    fig.text(0.5, 0.055, "Scores from pooled contingency counts; lines connect evaluated thresholds only.",
             ha="center", fontsize=9)
    save_figure(fig, args.output_dir, "group1_event_detection")
    write_plot_metadata(args.output_dir, files, manifest, __file__,
                        daily_difference_definition="method minus bilinear ERA5, paired by date",
                        box_whiskers="1.5 IQR; all fliers shown", uncertainty_intervals="not calculated",
                        daily_difference_summary=summary)
    print(f"Saved both figures as PNG and PDF in {args.output_dir}")


if __name__ == "__main__":
    main()
