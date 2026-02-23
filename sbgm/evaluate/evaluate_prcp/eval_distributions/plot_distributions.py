# sbgm/evaluate/evaluate_prcp/eval_distributional/plot_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _nice, _savefig
from sbgm.variable_utils import get_color_for_model

# Baseline overlays
from sbgm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs, load_csv_if_exists

# Helper to sanitize arrays for plotting (for log plots)
def _finite(arr: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Replace non-finite values with small positive floor (for log plots)."""
    return np.where(np.isfinite(arr), arr, floor)

logger = logging.getLogger(__name__)



SET_DPI = 300

# Z-order scheme so ensemble is on top, then HR, PMM, LR, then baselines
ZORDER_ENS = 25
ZORDER_HR = 20
ZORDER_PMM = 18
ZORDER_LR = 10
ZORDER_BASELINE = 5

# Default visualization toggles (can be overridden from eval_cfg)
DEFAULT_SHOW_CI = False    # Turn CI bands off by default
DEFAULT_SHOW_INSET = True  # Show tail-zoom inset by default
Y_FLOOR = 1e-7             # Log-axis floor to avoid collapsing


def _load_bins(tables: Path) -> Optional[np.ndarray]:
    bins_path = tables / "dist_bins.csv"
    if not bins_path.exists():
        return None
    bins = (
        np.loadtxt(bins_path, delimiter=",", skiprows=1)
        if bins_path.read_text().startswith("bin_edge")
        else np.loadtxt(bins_path, delimiter=",")
    )
    if bins.ndim > 1:
        bins = bins[:, 0]
    return bins


def _read_hist(tables: Path, name: str) -> Optional[np.ndarray]:
    p = tables / f"dist_{name}.csv"
    if not p.exists():
        return None
    xs, cs = [], []
    with open(p, "r") as f:
        next(f)  # header
        for ln in f:
            s = ln.strip().split(",")
            if len(s) != 2:
                continue
            xs.append(int(s[0]))
            cs.append(int(float(s[1])))
    return np.array(cs, dtype=float)


def _load_ensemble_artifacts(tables: Path):
    """Return (ens_mode, gen_ens_pool, gen_ens_mean, q10, q50, q90)."""
    ens_mode = None
    gen_ens_pool = None
    gen_ens_mean = None

    ens_npz = tables / "dist_member_histograms.npz"

    if (tables / "dist_gen_ens_pool.csv").exists():
        cs = []
        with open(tables / "dist_gen_ens_pool.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    cs.append(float(s[1]))
        gen_ens_pool = np.array(cs, dtype=float)
        ens_mode = "pool"

    if (tables / "dist_gen_ens_mean.csv").exists():
        ps = []
        with open(tables / "dist_gen_ens_mean.csv", "r") as f:
            next(f)
            for ln in f:
                s = ln.strip().split(",")
                if len(s) == 2:
                    ps.append(float(s[1]))
        gen_ens_mean = np.array(ps, dtype=float)
        ens_mode = "member_mean"

    q10 = q50 = q90 = None
    if ens_npz.exists():
        try:
            d = np.load(ens_npz)
            q10 = d.get("pdf_q10", None)
            q50 = d.get("pdf_q50", None)
            q90 = d.get("pdf_q90", None)
            if ens_mode is None and "mode" in d:
                try:
                    ens_mode = str(d["mode"])  # may be 0-d array
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"[plot_distributional] Could not load ensemble NPZ: {e}")

    return ens_mode, gen_ens_pool, gen_ens_mean, q10, q50, q90


def _load_metrics_text(tables: Path):
    """Return (gen_text, lr_text) for plot annotations."""
    metrics_path = tables / "dist_metrics.csv"
    gen_text = None
    lr_text = None
    if not metrics_path.exists():
        return gen_text, lr_text

    lines = metrics_path.read_text().strip().splitlines()
    rows = []
    for ln in lines[1:]:
        try:
            ref, comp, w1, ks_s, ks_p, kl = ln.split(",")
            rows.append((ref, comp, float(w1), float(ks_s), float(ks_p), float(kl)))
        except Exception:
            continue

    gen_parts = []
    lr_parts = []
    for (ref, comp, w1, kss, ksp, kl) in rows:
        if comp.lower() in ("gen_ens_pool", "gen_ens_mean", "gen_pmm"):
            txt = (
                f"{comp.upper()} vs {ref.upper()}:\n"
                f"  W1  = {w1:.3f}\n"
                f"  KS  = {kss:.3f}\n"
                f"  KL  = {kl:.3f}"
            )
            gen_parts.append(txt)
        if comp.lower() == "lr":
            txt = (
                f"{comp.upper()} vs {ref.upper()}:\n"
                f"  W1  = {w1:.3f}\n"
                f"  KS  = {kss:.3f} (p={ksp:.2f})\n"
                f"  KL  = {kl:.3f}"
            )
            lr_parts.append(txt)

    gen_text = "\n".join(gen_parts).strip() if gen_parts else None
    lr_text = "\n".join(lr_parts).strip() if lr_parts else None
    return gen_text, lr_text


def plot_pooled_distribution(
    dist_root: str | Path,
    eval_cfg: Any | None = None,
) -> None:
    """Pooled (all-days) 1D distribution plot.

    Dependencies:
      - tables/dist_bins.csv
      - tables/dist_hr.csv and/or tables/dist_gen.csv (LR optional)
      - tables/dist_metrics.csv (optional, for annotation)
      - tables/dist_gen_ens_pool.csv or tables/dist_gen_ens_mean.csv (optional)
      - tables/dist_member_histograms.npz (optional, for q10/q90 band)
      - tables/dist_daily.npz (optional, for CI bands + Npix)
    """
    dist_root = Path(dist_root)
    tables = dist_root / "tables"
    figs = _ensure_dir(dist_root / "figures")

    # Set colors
    col_hr = get_color_for_model("hr")
    col_pmm = get_color_for_model("pmm")
    col_ens = get_color_for_model("ensemble")
    col_lr = get_color_for_model("lr")

    bins = _load_bins(tables)
    if bins is None:
        logger.warning("[plot_pooled_distribution] No dist_bins.csv - skipping pooled plot.")
        return

    hr = _read_hist(tables, "hr")
    gen = _read_hist(tables, "gen")
    lr = _read_hist(tables, "lr")

    ens_mode, gen_ens_pool, gen_ens_mean, q10, q50, q90 = _load_ensemble_artifacts(tables)
    gen_text, lr_text = _load_metrics_text(tables)

    _nice()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    eps = 1e-12
    y_floor = Y_FLOOR

    def _norm(h: np.ndarray | None) -> np.ndarray | None:
        if h is None:
            return None
        s = h.sum()
        if s <= 0:
            return h
        return h / s

    hr_n = _norm(hr)
    gen_n = _norm(gen)
    lr_n = _norm(lr)

    # Optional toggles from eval config
    show_ci = DEFAULT_SHOW_CI
    show_inset = DEFAULT_SHOW_INSET
    if eval_cfg is not None:
        try:
            show_ci = bool(getattr(eval_cfg, "dist_show_ci", DEFAULT_SHOW_CI))
            show_inset = bool(getattr(eval_cfg, "dist_show_inset", DEFAULT_SHOW_INSET))
        except Exception:
            pass

    # Optional daily uncertainty bands
    daily_npz = tables / "dist_daily.npz"
    ci = {}
    if daily_npz.exists():
        try:
            npz = np.load(daily_npz)

            def _series_ci(counts_key: str, n_key: str):
                keys = getattr(npz, "files", None)
                if keys is None:
                    keys_attr = getattr(npz, "keys", None)
                    if callable(keys_attr):
                        try:
                            k = keys_attr()
                            try:
                                from collections.abc import Iterable

                                if isinstance(k, Iterable):
                                    keys = list(k)
                                else:
                                    keys = None
                            except Exception:
                                try:
                                    keys = list(k)  # type: ignore[arg-type]
                                except Exception:
                                    keys = None
                        except Exception:
                            keys = None
                    else:
                        keys = None

                if keys is None or (counts_key not in keys) or (n_key not in keys):
                    return None
                C = npz[counts_key]  # [D,B]
                n = npz[n_key]  # [D]
                if C.size == 0 or n.size == 0:
                    return None
                n = np.maximum(n.astype(float), 1.0)
                pdf = (C.astype(float).T / n).T  # [D,B]
                lo = np.percentile(pdf, 5, axis=0)
                hi = np.percentile(pdf, 95, axis=0)
                med = np.percentile(pdf, 50, axis=0)
                return lo, hi, med

            ci["hr"] = _series_ci("counts_hr", "n_hr")
            ci["gen"] = _series_ci("counts_gen", "n_gen")
            if "counts_lr" in npz and "n_lr" in npz:
                ci["lr"] = _series_ci("counts_lr", "n_lr")
        except Exception as e:
            logger.warning(f"[plot_pooled_distribution] Failed to parse dist_daily.npz for CI shading: {e}")

    if show_ci:
        val = ci.get("hr")
        if val is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_hr,
                alpha=0.10,
                linewidth=0,
            )
        val = ci.get("gen")
        if val is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_pmm,
                alpha=0.08,
                linewidth=0,
            )
        val = ci.get("lr")
        if val is not None and lr_n is not None:
            lo, hi, _ = val
            ax.fill_between(
                bin_centers,
                np.maximum(_finite(lo, eps), y_floor).tolist(),
                np.maximum(_finite(hi, eps), y_floor).tolist(),
                color=col_lr,
                alpha=0.07,
                linewidth=0,
            )

    def _percentile_from_hist(counts: np.ndarray | None, bins_arr: np.ndarray, p: float) -> Optional[float]:
        if counts is None or counts.size == 0:
            return None
        c = np.cumsum(counts.astype(float))
        c /= max(c[-1], 1.0)
        idx = np.searchsorted(c, p)
        idx = int(np.clip(idx, 0, len(bins_arr) - 2))
        return float(0.5 * (bins_arr[idx] + bins_arr[idx + 1]))

    if lr_n is not None:
        ax.plot(
            bin_centers,
            lr_n,
            color=col_lr,
            lw=1.0,
            ls="--",
            label="LR",
            zorder=ZORDER_LR,
        )
    if hr_n is not None:
        ax.plot(
            bin_centers,
            hr_n,
            color=col_hr,
            lw=1.5,
            label="HR",
            zorder=ZORDER_HR,
        )
    if gen_n is not None:
        ax.plot(
            bin_centers,
            gen_n,
            color=col_pmm,
            ls="-.",
            lw=1.2,
            label="PMM",
            zorder=ZORDER_PMM,
        )
    ens_curve = None
    if gen_ens_pool is not None:
        s = float(np.sum(gen_ens_pool))
        if s > 0:
            ens_curve = gen_ens_pool / s
    elif gen_ens_mean is not None:
        ens_curve = np.asarray(gen_ens_mean, dtype=float)

    # Reference wet-day threshold (kept for optional debugging/annotation)
    _ = float(getattr(eval_cfg, "wet_threshold_mm", 1.0)) if eval_cfg is not None else 1.0

    p95 = _percentile_from_hist(hr, bins, 0.95)
    p99 = _percentile_from_hist(hr, bins, 0.99)
    p999 = _percentile_from_hist(hr, bins, 0.999)
    p9999 = _percentile_from_hist(hr, bins, 0.9999)
    p99999 = _percentile_from_hist(hr, bins, 0.99999)

    ylim_main = ax.get_ylim()
    y_ann = ylim_main[1] * 0.6
    if p95 is not None:
        ax.axvline(p95, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p95, y_ann, "P95", rotation=90, va="top", ha="right", fontsize=8, color="0.25")
    if p99 is not None:
        ax.axvline(p99, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p99, y_ann, "P99", rotation=90, va="top", ha="right", fontsize=8, color="0.25")
    if p999 is not None:
        ax.axvline(p999, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p999, y_ann, "P99.9", rotation=90, va="top", ha="right", fontsize=8, color="0.25")
    if p9999 is not None:
        ax.axvline(p9999, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p9999, y_ann, "P99.99", rotation=90, va="top", ha="right", fontsize=8, color="0.25")
    if p99999 is not None:
        ax.axvline(p99999, color="0.2", lw=0.8, ls="--", alpha=0.6)
        ax.text(p99999, y_ann, "P99.999", rotation=90, va="top", ha="right", fontsize=8, color="0.25")

    # Tail inset (optional)
    try:
        if show_inset:
            x_min = 20.0
            x_max = min(80.0, float(bins.max()))
            if x_max > x_min:
                ax_ins = inset_axes(
                    ax,
                    width="36%",
                    height="56%",
                    loc="upper right",
                    bbox_to_anchor=(-0.22, 1.0),
                    bbox_transform=ax.transAxes,
                    borderpad=0.0,
                )
                if hr_n is not None:
                    ax_ins.plot(bin_centers, hr_n, color=col_hr, lw=1.2)
                if gen_n is not None:
                    ax_ins.plot(bin_centers, gen_n, color=col_pmm, lw=1.0)
                if lr_n is not None:
                    ax_ins.plot(bin_centers, lr_n, color=col_lr, lw=0.9, ls="--")
                if ens_curve is not None:
                    ax_ins.plot(bin_centers, np.maximum(ens_curve, eps), lw=1.0, color=col_ens)
                ax_ins.set_xlim(x_min, x_max)
                ax_ins.set_yscale("log")
                ax_ins.set_ylim(max(ax.get_ylim()[0], y_floor), ax.get_ylim()[1])
                ax_ins.tick_params(labelsize=7)
                ax_ins.grid(True, ls=":", alpha=0.3)
    except Exception as e:
        logger.info(f"[plot_pooled_distribution] Tail inset skipped: {e}")

    if ens_curve is not None:
        ax.plot(
            bin_centers,
            np.maximum(ens_curve, eps),
            lw=1.6,
            label="GEN (ensemble)",
            color=col_ens,
            zorder=ZORDER_ENS,
        )
    if q10 is not None and q90 is not None:
        ax.fill_between(
            bin_centers,
            np.maximum(q10, eps).tolist(),
            np.maximum(q90, eps).tolist(),
            alpha=0.10,
            linewidth=0,
            label="Ens spread (10-90%)",
        )

    # === Baseline overlays ===
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo:
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="distributional",
            )
        except Exception as e:
            logger.warning(f"[plot_pooled_distribution] Failed to resolve baseline dirs: {e}")
            dirs = {}
        for t, d0 in dirs.items():
            try:
                bins_arr = None
                bins_path_b = d0 / "dist_bins.csv"
                if bins_path_b.exists():
                    bins_arr = (
                        np.loadtxt(bins_path_b, delimiter=",", skiprows=1)
                        if bins_path_b.read_text().startswith("bin_edge")
                        else np.loadtxt(bins_path_b, delimiter=",")
                    )
                    if bins_arr.ndim > 1:
                        bins_arr = bins_arr[:, 0]
                else:
                    logger.info(f"[plot_pooled_distribution] Baseline {t}: missing dist_bins.csv at {bins_path_b}")
                    continue

                bin_centers_b = 0.5 * (bins_arr[:-1] + bins_arr[1:])
                arr = load_csv_if_exists(d0, "dist_gen")
                if arr is None:
                    arr = load_csv_if_exists(d0, "dist_lr")
                    if arr is None:
                        logger.info(
                            f"[plot_pooled_distribution] Baseline {t}: missing both dist_gen.csv and dist_lr.csv in {d0}"
                        )
                        continue


                try:
                    counts = np.asarray(arr["count"], dtype=float)
                except Exception:
                    counts = np.asarray(arr[:, 1], dtype=float)
                pdf = counts / (np.sum(counts) + eps)
                label = bo.get("labels", {}).get(t, t)
                style = dict(bo.get("styles", {}).get(t, {}))
                style.setdefault("zorder", ZORDER_BASELINE)
                ax.plot(bin_centers_b, pdf, label=label, **style)
            except Exception as e:
                logger.info(f"[plot_pooled_distribution] Baseline overlay for {t} failed: {e}")
                continue

    ax.set_xlabel("Precipitation (mm/day)")
    ax.set_yscale("log")
    ax.set_ylabel("Probability")
    ax.set_title("Pooled pixel distributions", fontsize=15, pad=10)

    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], y_floor))

    boxprops = dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85)
    if gen_text:
        ax.text(
            0.02,
            0.4,
            gen_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=boxprops,
        )
    if lr_text:
        ax.text(
            0.02,
            0.02,
            lr_text,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8,
            bbox=boxprops,
        )

    try:
        _savefig(fig, figs / "dist_pooled.png", dpi=SET_DPI)
    except Exception as e:
        logger.warning(f"[plot_pooled_distribution] Failed saving pooled plot: {e}")
    plt.close(fig)


def plot_seasonal_distributions(
    dist_root: str | Path,
    eval_cfg: Any | None = None,
) -> None:
    """Seasonal distributions plot.

    Dependencies:
      - tables/dist_daily.npz (required)
      - tables/dist_bins.csv (fallback bins if daily.npz lacks them)

    Note: seasonal plots intentionally do NOT auto-trigger pooled plotting.
    """
    dist_root = Path(dist_root)
    tables = dist_root / "tables"
    figs = _ensure_dir(dist_root / "figures")

    daily_npz = tables / "dist_daily.npz"
    if not daily_npz.exists():
        logger.warning("[plot_seasonal_distributions] dist_daily.npz missing - cannot plot seasons. Run daily_hist first.")
        return

    # Set colors
    col_hr = get_color_for_model("hr")
    col_pmm = get_color_for_model("pmm")
    col_lr = get_color_for_model("lr")

    bins_fallback = _load_bins(tables)
    if bins_fallback is None:
        logger.warning("[plot_seasonal_distributions] dist_bins.csv missing; will require bins inside dist_daily.npz.")

    try:
        d = np.load(daily_npz)
        bins_s = d["bins"] if "bins" in d else bins_fallback
        if bins_s is None:
            logger.warning("[plot_seasonal_distributions] No bins available - skipping seasonal plot.")
            return
        mids_s = 0.5 * (bins_s[:-1] + bins_s[1:])

        dates_s = d["dates"].astype(str) if "dates" in d else np.array([], dtype=str)

        def _season(yyyymmdd: str) -> str:
            try:
                m = datetime.strptime(yyyymmdd, "%Y%m%d").month
            except Exception:
                return "UNK"
            if m in (12, 1, 2):
                return "DJF"
            if m in (3, 4, 5):
                return "MAM"
            if m in (6, 7, 8):
                return "JJA"
            return "SON"

        seasons = np.array([_season(s) for s in dates_s], dtype="U")

        # Required arrays
        counts_hr = d.get("counts_hr", None)
        counts_gen = d.get("counts_gen", None)
        n_hr = d.get("n_hr", None)
        n_gen = d.get("n_gen", None)

        if counts_hr is None or counts_gen is None or n_hr is None or n_gen is None:
            logger.warning("[plot_seasonal_distributions] dist_daily.npz missing required keys (counts_hr/counts_gen/n_hr/n_gen).")
            return

        counts_lr = d.get("counts_lr", None)
        n_lr = d.get("n_lr", None)

        _nice()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        ax_map = {"DJF": axes[0, 0], "MAM": axes[0, 1], "JJA": axes[1, 0], "SON": axes[1, 1]}

        eps = 1e-12
        y_floor = Y_FLOOR

        def _pool_pdf(C: np.ndarray, n: np.ndarray, sel: np.ndarray) -> Optional[np.ndarray]:
            if C is None or n is None:
                return None
            if C.size == 0 or n.size == 0:
                return None
            if np.sum(sel) == 0:
                return None
            C_sel = C[sel]
            n_sel = np.maximum(n[sel].astype(float), 1.0)
            # pooled pdf across selected days
            pooled_counts = C_sel.sum(axis=0).astype(float)
            pooled_pdf = pooled_counts / max(pooled_counts.sum(), 1.0)
            return pooled_pdf

        for sname in ("DJF", "MAM", "JJA", "SON"):
            ax = ax_map[sname]
            sel = (seasons == sname)

            hr_pdf = _pool_pdf(counts_hr, n_hr, sel)
            gen_pdf = _pool_pdf(counts_gen, n_gen, sel)
            lr_pdf = _pool_pdf(counts_lr, n_lr, sel) if (counts_lr is not None and n_lr is not None) else None

            if lr_pdf is not None:
                ax.plot(mids_s, np.maximum(lr_pdf, eps), color=col_lr, lw=1.0, ls="--", label="LR")
            if hr_pdf is not None:
                ax.plot(mids_s, np.maximum(hr_pdf, eps), color=col_hr, lw=1.5, label="HR")
            if gen_pdf is not None:
                ax.plot(mids_s, np.maximum(gen_pdf, eps), color=col_pmm, lw=1.2, ls="-.", label="PMM")

            ax.set_title(sname)
            ax.grid(True, ls=":", alpha=0.4)
            ax.set_yscale("log")
            ax.set_ylim(bottom=y_floor)

        # Shared labels
        for ax in axes[1, :]:
            ax.set_xlabel("Precipitation (mm/day)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Probability")

        # One legend (top-left)
        axes[0, 0].legend(fontsize=9)

        fig.suptitle("Seasonal pooled pixel distributions", fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        try:
            _savefig(fig, figs / "dist_seasons.png", dpi=SET_DPI)
        except Exception as e:
            logger.warning(f"[plot_seasonal_distributions] Failed saving seasonal plot: {e}")
        plt.close(fig)

    except Exception as e:
        logger.warning(f"[plot_seasonal_distributions] Failed to build seasonal plots: {e}")


def plot_distributional(
    dist_root: str | Path,
    eval_cfg: Any | None = None,
    *,
    plot_pooled: bool = True,
    plot_seasons: bool = True,
) -> None:
    """Backward-compatible wrapper.

    IMPORTANT: tasks should call `plot_pooled_distribution` and/or
    `plot_seasonal_distributions` directly to avoid surprises.
    """
    if plot_pooled:
        plot_pooled_distribution(dist_root, eval_cfg=eval_cfg)
    if plot_seasons:
        plot_seasonal_distributions(dist_root, eval_cfg=eval_cfg)