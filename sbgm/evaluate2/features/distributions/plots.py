from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sbgm.evaluate2.config import Eval2Plan
from sbgm.evaluate2.store import FeatureStore
from sbgm.evaluate2.features.distributions.feature import DistributionsConfig

logger = logging.getLogger(__name__)


def _load_npz(path: Path):
    return np.load(path, allow_pickle=True)


def _pdf(counts: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return counts.astype(np.float64) / float(n)


def _cdf_from_counts(counts: np.ndarray) -> np.ndarray:
    s = float(np.sum(counts))
    if s <= 0:
        return np.zeros_like(counts, dtype=np.float64)
    return np.cumsum(counts.astype(np.float64) / s)


def _percentile_from_hist(centers: np.ndarray, counts: np.ndarray, q: float) -> Optional[float]:
    if counts.size == 0 or np.sum(counts) <= 0:
        return None
    cdf = _cdf_from_counts(counts)
    target = float(q) / 100.0
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(max(idx, 0), len(centers) - 1)
    return float(centers[idx])


def _ci_from_daily(daily_counts: np.ndarray, daily_n: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if daily_counts.ndim != 2 or daily_counts.shape[0] == 0:
        return None
    pdfs = []
    for c, n in zip(daily_counts, daily_n):
        if int(n) <= 0:
            continue
        pdfs.append(c.astype(np.float64) / float(n))
    if not pdfs:
        return None
    A = np.stack(pdfs, axis=0)
    q10 = np.nanquantile(A, 0.10, axis=0)
    q90 = np.nanquantile(A, 0.90, axis=0)
    return q10, q90


def _parse_month(date_str: str) -> int:
    if "-" in date_str:
        parts = date_str.split("-")
        if len(parts) >= 2:
            return int(parts[1])
    else:
        if len(date_str) >= 6:
            return int(date_str[4:6])
    raise ValueError(f"Unrecognized date format: {date_str}")


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    if m in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid month: {m}")


def _load_metrics_text(metrics_path: Path) -> Optional[str]:
    if not metrics_path.exists():
        return None
    rows = []
    try:
        with metrics_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.warning("[eval2:distributions] failed to read %s: %s", metrics_path, e)
        return None

    if not rows:
        return None

    blocks = []
    for row in rows:
        comp = str(row.get("comp", "?")).upper()
        try:
            w1 = float(row.get("w1", "nan"))
            ks = float(row.get("ks", "nan"))
            kl = float(row.get("kl", "nan"))
        except Exception:
            continue
        blocks.append(f"{comp} vs HR:\nW1 = {w1:.3f}\nKS = {ks:.3f}\nKL = {kl:.3f}")
    return "\n\n".join(blocks) if blocks else None


def _add_percentile_lines(ax, centers: np.ndarray, counts_hr: np.ndarray) -> None:
    levels = [95.0, 99.0, 99.9, 99.99]
    for q in levels:
        x = _percentile_from_hist(centers, counts_hr, q)
        if x is None:
            continue
        ax.axvline(x, ls="--", lw=0.9, alpha=0.45)
        ax.text(
            x,
            0.95,
            f"P{q:g}",
            rotation=90,
            va="top",
            ha="right",
            transform=ax.get_xaxis_transform(),
            alpha=0.7,
        )


def _style_axis(ax, title: str) -> None:
    ax.set_yscale("log")
    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def plot_distributions(plan: Eval2Plan, store: FeatureStore, cfg: DistributionsConfig) -> None:
    """Plot pooled and optional seasonal distributions from saved tables.

    Constraints:
      - Must not touch resolver.
      - Must gracefully skip if required artifacts are missing.
    """

    bins_path = store.path_table("bins.npz")
    pooled_path = store.path_table("pooled_counts.npz")

    if not bins_path.exists() or not pooled_path.exists():
        logger.warning(
            "[eval2:distributions] plot requested but required tables missing (bins=%s pooled=%s); skipping",
            bins_path,
            pooled_path,
        )
        return

    b = _load_npz(bins_path)
    centers = np.asarray(b["centers"], dtype=np.float64)

    p = _load_npz(pooled_path)
    c_hr = np.asarray(p["counts_hr"], dtype=np.int64)
    n_hr = int(np.asarray(p["n_hr"]).item())
    c_gen = np.asarray(p.get("counts_gen", np.zeros_like(c_hr)), dtype=np.int64)
    n_gen = int(np.asarray(p.get("n_gen", 0)).item())

    has_lr = "counts_lr" in p.files
    if has_lr:
        c_lr = np.asarray(p["counts_lr"], dtype=np.int64)
        n_lr = int(np.asarray(p["n_lr"]).item())
    else:
        c_lr = None
        n_lr = 0

    pdf_hr = _pdf(c_hr, n_hr)
    pdf_gen = _pdf(c_gen, n_gen)

    ci_hr = None
    ci_gen = None
    if cfg.plot_ci_daily:
        daily_path = store.path_table("daily_counts.npz")
        if daily_path.exists():
            d = _load_npz(daily_path)
            if "counts_hr" in d.files and "n_hr" in d.files:
                ci_hr = _ci_from_daily(np.asarray(d["counts_hr"], dtype=np.int64), np.asarray(d["n_hr"], dtype=np.int64))
            if "counts_gen" in d.files and "n_gen" in d.files:
                ci_gen = _ci_from_daily(np.asarray(d["counts_gen"], dtype=np.int64), np.asarray(d["n_gen"], dtype=np.int64))

    if cfg.plot_pooled:
        fig = plt.figure(figsize=(9.0, 6.2))
        ax = fig.add_subplot(111)
        _style_axis(ax, "Pooled pixel distributions")

        ax.plot(centers, pdf_hr + 1e-18, label="HR", lw=1.6)
        ax.plot(centers, pdf_gen + 1e-18, label="GEN", lw=1.6)

        if ci_hr is not None:
            ax.fill_between(centers, ci_hr[0] + 1e-18, ci_hr[1] + 1e-18, alpha=0.18)
        if ci_gen is not None:
            ax.fill_between(centers, ci_gen[0] + 1e-18, ci_gen[1] + 1e-18, alpha=0.18)

        if c_lr is not None and n_lr > 0:
            ax.plot(centers, _pdf(c_lr, n_lr) + 1e-18, label="LR", lw=1.4, ls="--")

        if cfg.plot_percentile_lines:
            _add_percentile_lines(ax, centers, c_hr)

        if cfg.plot_metrics_box:
            metrics_text = _load_metrics_text(store.path_table("metrics.csv"))
            if metrics_text:
                ax.text(
                    0.02,
                    0.03,
                    metrics_text,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

        ax.legend(loc="upper right")
        out_path = store.path_figure("pooled.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("[eval2:distributions] wrote %s", out_path)

    if cfg.plot_seasonal:
        daily_path = store.path_table("daily_counts.npz")
        if not daily_path.exists():
            logger.warning("[eval2:distributions] plot_seasonal=True but daily_counts.npz missing; skipping seasonal plots")
            return

        d = _load_npz(daily_path)
        if "dates" not in d.files:
            logger.warning("[eval2:distributions] daily_counts.npz missing 'dates' array; skipping seasonal plots")
            return

        dates = d["dates"]
        counts_hr_d = np.asarray(d["counts_hr"], dtype=np.int64)
        n_hr_d = np.asarray(d["n_hr"], dtype=np.int64)
        counts_gen_d = np.asarray(d["counts_gen"], dtype=np.int64)
        n_gen_d = np.asarray(d["n_gen"], dtype=np.int64)

        has_lr_d = "counts_lr" in d.files and "n_lr" in d.files
        if has_lr_d:
            counts_lr_d = np.asarray(d["counts_lr"], dtype=np.int64)
            n_lr_d = np.asarray(d["n_lr"], dtype=np.int64)
        else:
            counts_lr_d = None
            n_lr_d = None

        season_indices = {"DJF": [], "MAM": [], "JJA": [], "SON": []}
        for i, date_raw in enumerate(dates):
            date_str = date_raw.decode("utf-8") if isinstance(date_raw, (bytes, np.bytes_)) else str(date_raw)
            try:
                season_indices[_season_from_month(_parse_month(date_str))].append(i)
            except Exception as ex:
                logger.warning("[eval2:distributions] skipping date %s due to parsing error: %s", date_str, ex)

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)
        season_order = ["DJF", "MAM", "JJA", "SON"]
        bg = {"DJF": "#dfe8f4", "MAM": "#e8f0e5", "JJA": "#f4efcf", "SON": "#f4e6e3"}

        legend_handles = None
        legend_labels = None
        for ax, season in zip(axes.flat, season_order):
            ax.set_facecolor(bg[season])
            _style_axis(ax, season)
            idxs = season_indices[season]
            if not idxs:
                continue

            c_hr_s = np.sum(counts_hr_d[idxs, :], axis=0)
            n_hr_s = int(np.sum(n_hr_d[idxs]))
            c_gen_s = np.sum(counts_gen_d[idxs, :], axis=0)
            n_gen_s = int(np.sum(n_gen_d[idxs]))

            if n_hr_s > 0:
                ax.plot(centers, _pdf(c_hr_s, n_hr_s) + 1e-18, label="HR", lw=1.6)
            if n_gen_s > 0:
                ax.plot(centers, _pdf(c_gen_s, n_gen_s) + 1e-18, label="GEN", lw=1.6)

            if counts_lr_d is not None and n_lr_d is not None:
                c_lr_s = np.sum(counts_lr_d[idxs, :], axis=0)
                n_lr_s = int(np.sum(n_lr_d[idxs]))
                if n_lr_s > 0:
                    ax.plot(centers, _pdf(c_lr_s, n_lr_s) + 1e-18, label="LR", lw=1.4, ls="--")

            if cfg.plot_percentile_lines:
                _add_percentile_lines(ax, centers, c_hr_s)

            legend_handles, legend_labels = ax.get_legend_handles_labels()

        if legend_handles and legend_labels:
            fig.legend(legend_handles, legend_labels, loc="center right", frameon=True)
        fig.tight_layout(rect=(0.0, 0.0, 0.88, 1.0))
        out_season = store.path_figure("seasonal.png")
        out_season.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_season, dpi=200)
        plt.close(fig)
        logger.info("[eval2:distributions] wrote %s", out_season)
