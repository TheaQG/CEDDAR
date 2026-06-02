from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Any, List

import numpy as np
import matplotlib.pyplot as plt
import logging

from sbgm.evaluate.evaluate_prcp.plot_utils import _ensure_dir, _savefig, _nice, _to_date_safe, _season_from_month
from sbgm.variable_utils import get_units, get_color_for_model, get_cmap_for_variable
from sbgm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs

logger = logging.getLogger(__name__)

SET_DPI = 300
# Get colors and set them
COL_HR = get_color_for_model("HR")
COL_PMM = get_color_for_model("PMM")
COL_LR = get_color_for_model("LR")
COL_ENS = get_color_for_model("ensemble")
COL_QM = get_color_for_model("QM")
COL_UNET = get_color_for_model("unet")

# Preferred drawing order (larger zorder = on top)
ZORDER_HR = 20          # HR DANRA
ZORDER_PMM = 18         # PMM (gen)
ZORDER_ENS = 25         # GEN ensemble mean
ZORDER_LR = 10          # LR
ZORDER_BASELINE = 5     # extra baselines (QM, UNet-SR, etc.)
ZORDER_ANNOT = 2        # vertical lines, text, etc.

# ================================================================================
# 1. PSD curves
# ================================================================================

def plot_scale_psd(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Read scale_psd_curves.npz and make a single log-log PSD plot:
      - HR: thick, solid
      - GEN: solid
      - LR: full curve, but 'ghosted' (low alpha) for k < lr_nyquist and solid for k >= lr_nyquist
      - x-axis in wavelength lambda (km), log-scaled, inverted (large -> small)
      - HR: thick, black, with +/- 1 sigma shading
      - GEN/PMM: blue, with +/- 1 sigma shading
      - LR: pink
          - solid for λ >= λ_nyq   (i.e. k <= k_nyq)
          - faint/dashed for λ < λ_nyq (i.e. k > k_nyq)
      - vertical line at LR Nyquist
      - optional amplitude alignment for LR so it sits in the right ballpark
        compared to HR on the trusted (k <= k_nyq) range
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")

    npz_path = tables / "scale_psd_curves.npz"
    if not npz_path.exists():
        logger.warning(f"[plot_scale_psd] Did not find {npz_path} - skipping PSD plot.")
        return

    # Set colors
    col_hr = get_color_for_model("HR")
    col_gen = get_color_for_model("PMM")
    col_lr = get_color_for_model("LR")
    col_gen_ens = get_color_for_model("ensemble")

    with np.load(npz_path) as data:
        files = set(data.files)
        def _opt(key):
            return data[key] if key in files else None
        k = data["k"]               # [K]
        psd_hr = data["psd_hr"]     # [N, K]
        psd_gen = data["psd_gen"]   # [N, K]
        psd_lr = data["psd_lr"]     # [N, K]
        psd_lr_hr = data["psd_lr_hr"]  # [N, K]
        dates = data["dates"]          # [N]
        psd_hr_ci_lo = _opt("psd_hr_ci_lo")
        psd_hr_ci_hi = _opt("psd_hr_ci_hi")
        psd_gen_ci_lo = _opt("psd_gen_ci_lo")
        psd_gen_ci_hi = _opt("psd_gen_ci_hi")
        lr_nyquist_arr = _opt("lr_nyquist")
        lr_nyquist = float(lr_nyquist_arr) if lr_nyquist_arr is not None else 0.0
        # default low-k upper bound (k <= 1/200 km), but allow extending up to LR Nyquist if available
        low_k_max_default = 1.0 / 200.0
        low_k_max_eff = lr_nyquist if lr_nyquist > 0.0 else low_k_max_default        
        psd_gen_ens_mean = _opt("psd_gen_ens_mean")
        psd_gen_ens_ci_lo = _opt("psd_gen_ens_ci_lo")
        psd_gen_ens_ci_hi = _opt("psd_gen_ens_ci_hi")

    if psd_gen_ens_mean is None:
        logger.info("[plot_scale_psd] No ensemble PSD arrays found in NPZ -> only PMM will be plotted.")

    # mean over dates
    eps = 1e-12
    hr_mean = psd_hr.mean(axis=0)
    hr_std = psd_hr.std(axis=0)
    gen_mean = psd_gen.mean(axis=0)
    gen_std = psd_gen.std(axis=0)
    lr_mean = psd_lr.mean(axis=0)

    lr_hr_mean = None
    if psd_lr_hr is not None:
        lr_hr_mean = psd_lr_hr.mean(axis=0)

    hr_mean = np.maximum(hr_mean, eps)
    gen_mean = np.maximum(gen_mean, eps)
    lr_mean = np.maximum(lr_mean, eps)


    if lr_nyquist > 0.0:
        lr_mask_lo = k <= lr_nyquist * 1.0001
        lr_mask_hi = k > lr_nyquist * 1.0001
    else:
        # no LR Nyquist info → just plot as one line
        lr_mask_lo = np.ones_like(k, dtype=bool)
        lr_mask_hi = np.zeros_like(k, dtype=bool)

    # Convert to wavelength (km)
    mask_pos = k > 0.0
    k_pos = k[mask_pos]
    lam = 1.0 / k_pos
    # Sort from large to small wavelength so line is monotonic on x
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    hr_mean = hr_mean[mask_pos][order]
    gen_mean = gen_mean[mask_pos][order]
    hr_std = hr_std[mask_pos][order]
    gen_std = gen_std[mask_pos][order]
    lr_mean = lr_mean[mask_pos][order]
    if lr_hr_mean is not None:
        lr_hr_mean = np.maximum(lr_hr_mean, eps)
        lr_hr_mean = lr_hr_mean[mask_pos][order]

    gen_ens_mean = None
    gen_ens_ci_lo = None
    gen_ens_ci_hi = None
    if psd_gen_ens_mean is not None:
        arr = np.maximum(np.asarray(psd_gen_ens_mean), eps)[mask_pos][order]
        gen_ens_mean = arr
        if psd_gen_ens_ci_lo is not None and psd_gen_ens_ci_hi is not None:
            gen_ens_ci_lo = np.maximum(np.asarray(psd_gen_ens_ci_lo), eps)[mask_pos][order]
            gen_ens_ci_hi = np.maximum(np.asarray(psd_gen_ens_ci_hi), eps)[mask_pos][order]

    # --- Compute band powers and ratios from mean PSDs (as plotted) ---
    # 1. k-array in plotted order
    k_plot = k_pos[order]
    # 2. Band definitions
    # low-k goes all the way up to LR Nyquist if we know it; otherwise fall back to 1/200 km
    low_k_max = float(low_k_max_eff)              # k <= ...  -> λ >= ...
    high_k_min = 1.0 / 20.0                       # k >= 5.000e-02 -> λ <= 20 km
    # 3. Helper for band integration
    def _band_int(k_arr: np.ndarray, p_arr: np.ndarray, kmin: float, kmax: float) -> float:
        m = (k_arr >= kmin) & (k_arr <= kmax)
        if np.any(m):
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 1: requested band is entirely *below* available k -> use lowest bin
        if kmax < k_arr.min():  # e.g. low-k band but k_arr starts higher
            m = (k_arr <= k_arr.min() * 1.01)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Fallback 2: requested band is entirely *above* available k -> use highest bin
        if kmin > k_arr.max():  # e.g. very-high-k band
            m = (k_arr >= k_arr.max() * 0.99)
            return float(np.trapz(p_arr[m], k_arr[m]))

        # Final fallback - should rarely happen
        return float(np.trapz(p_arr, k_arr))

    # 4. Compute band powers for HR, GEN, LR-on-HR-grid (or native LR)
    hr_low  = _band_int(k_plot, hr_mean, 0.0, low_k_max)
    hr_high = _band_int(k_plot, hr_mean, high_k_min, k_plot.max())
    gen_low  = _band_int(k_plot, gen_mean, 0.0, low_k_max)
    gen_high = _band_int(k_plot, gen_mean, high_k_min, k_plot.max())
    if lr_hr_mean is not None:
        lr_used = lr_hr_mean
    else:
        lr_used = lr_mean
    lr_low  = _band_int(k_plot, lr_used, 0.0, low_k_max)
    lr_high = _band_int(k_plot, lr_used, high_k_min, k_plot.max())
    # 5. Compute ratios (guard against zero/NaN)
    def _safe_ratio(num: float, den: float, eps_d: float = 1e-12) -> float:
        if num is None or not np.isfinite(num) or num < 0.0:
            return float("nan")
        den_eff = den if (den is not None and np.isfinite(den) and den > 0.0) else eps_d
        return float(num / den_eff)
    gen_hr_low_ratio  = _safe_ratio(gen_low, hr_low)
    gen_hr_high_ratio = _safe_ratio(gen_high, hr_high)
    lr_hr_low_ratio   = _safe_ratio(lr_low, hr_low)
    lr_hr_high_ratio  = _safe_ratio(lr_high, hr_high)
    # Keep these ratios local to the plot for annotation only.

    # --- slope fits in log10(k) vs log10(P) ---
    def _fit_slope(k_arr: np.ndarray, p_arr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float, float, float]:
        xs = np.log10(k_arr[mask])
        ys = np.log10(np.maximum(p_arr[mask], eps))
        if xs.size < 2:
            return float("nan"), float("nan"), 0.0, float(np.nan), float(np.nan)
        b, a = np.polyfit(xs, ys, 1)  # y = b*x + a
        yhat = b * xs + a
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        return float(b), float(a), float(r2), float(k_arr[mask].min()), float(k_arr[mask].max())

    def _collect_slopes(name: str, p: np.ndarray) -> list[tuple[str, str, float, float, float, float, float]]:
        out_rows = []
        masks = {
            "all": np.isfinite(k_plot) & np.isfinite(p) & (p > 0),
            "low-k": (k_plot <= low_k_max) & np.isfinite(p) & (p > 0),
            "high-k": (k_plot >= high_k_min) & np.isfinite(p) & (p > 0),
        }
        for rng, m in masks.items():
            sl, ic, r2, kmin, kmax = _fit_slope(k_plot, p, m)
            out_rows.append((name, rng, sl, ic, r2, kmin, kmax))
        return out_rows

    slope_rows: list[tuple[str, str, float, float, float, float, float]] = []
    slope_rows += _collect_slopes("HR", hr_mean)
    slope_rows += _collect_slopes("GEN", gen_mean)  # PMM
    if gen_ens_mean is not None:
        slope_rows += _collect_slopes("GEN_ens", gen_ens_mean)  # ensemble mean
    if lr_hr_mean is not None:
        slope_rows += _collect_slopes("LR", lr_hr_mean)
    else:
        slope_rows += _collect_slopes("LR", lr_mean)

    # Keep these low/high-k slopes local to the plot for annotation only.


    # --- Find HR-LR intersection in log space ---
    cross_info = None
    if lr_hr_mean is not None:
        lr_for_x = lr_hr_mean.copy()
    else:
        lr_for_x = lr_mean.copy()
    try:
        diff = np.abs(np.log10(hr_mean) - np.log10(lr_for_x))
        ix = int(np.argmin(diff))
        lam_cross = float(lam[ix])
        k_cross = float(k_pos[order][ix])
        cross_info = (lam_cross, k_cross)
    except Exception:
        cross_info = None

    # Nyquist as wavelength
    lam_nyq = None
    if lr_nyquist > 0.0:
        lam_nyq = 1.0 / lr_nyquist

    _nice()
    # a bit wider so the legend can sit outside without crushing the axes
    fig, ax = plt.subplots(figsize=(7.1, 5.5))

    # HR
    ax.plot(
        lam, hr_mean,
        color=col_hr, lw=1.6, label="HR (DANRA)",
        zorder=ZORDER_HR,
    )
    if psd_hr_ci_lo is not None and psd_hr_ci_hi is not None:
        ci_lo = np.asarray(psd_hr_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_hr_ci_hi)[mask_pos][order]
        ax.fill_between(
            lam,
            np.maximum(ci_lo, eps), np.maximum(ci_hi, eps),  # type: ignore
            color=col_hr, alpha=0.15,
            zorder=ZORDER_HR - 1,
        )
    else:
        ax.fill_between(
            lam,
            np.maximum(hr_mean - hr_std, eps),
            hr_mean + hr_std,
            color=col_hr, alpha=0.15,
            zorder=ZORDER_HR - 1,
        )

    # GEN / PMM
    ax.plot(
        lam, gen_mean,
        color=col_gen, lw=1.4, label="PMM (gen)", ls='-.',
        zorder=ZORDER_PMM,
    )
    if psd_gen_ci_lo is not None and psd_gen_ci_hi is not None:
        ci_lo = np.asarray(psd_gen_ci_lo)[mask_pos][order]
        ci_hi = np.asarray(psd_gen_ci_hi)[mask_pos][order]
        ax.fill_between(
            lam,
            np.maximum(ci_lo, eps), np.maximum(ci_hi, eps),  # type: ignore
            color=col_gen, alpha=0.12,
            zorder=ZORDER_PMM - 1,
        )
    else:
        ax.fill_between(
            lam,
            np.maximum(gen_mean - gen_std, eps),
            gen_mean + gen_std,
            color=col_gen, alpha=0.12,
            zorder=ZORDER_PMM - 1,
        )

    # GEN ensemble mean + CI band (if available)
    if gen_ens_mean is not None:
        ax.plot(
            lam, gen_ens_mean,
            color=col_gen_ens, lw=1.2, label="GEN (ens mean)",
            zorder=ZORDER_ENS,
        )
        if gen_ens_ci_lo is not None and gen_ens_ci_hi is not None:
            ax.fill_between(
                lam, gen_ens_ci_lo, gen_ens_ci_hi, # type: ignore
                color=col_gen_ens, alpha=0.08,  
                zorder=ZORDER_ENS - 1,
            )

    # LR
    if lr_nyquist > 0.0 and lam_nyq is not None:
        trusted = lam >= (lam_nyq * 0.999)
        ghost   = lam <  (lam_nyq * 0.999)

        # "Ghost" part
        if lr_hr_mean is not None:
            ax.plot(
                lam, lr_hr_mean,
                color=col_lr, lw=0.9, linestyle="--", alpha=0.35,
                label="LR (ERA5, > Nyq)",
                zorder=ZORDER_LR - 1,
            )
        else:
            ax.plot(
                lam, lr_mean,
                color=col_lr, lw=0.9, linestyle="--", alpha=0.35,
                label="LR (ERA5, > Nyq)",
                zorder=ZORDER_LR - 1,
            )

        ax.axvline(
            x=lam_nyq, color="black", lw=0.6, linestyle="--",
            zorder=ZORDER_ANNOT, #label="LR Nyq",
        )

        # Solid, trusted LR (native spacing) on top of its ghost
        if np.any(trusted):
            if lr_hr_mean is not None:
                ax.plot(
                    lam[trusted], lr_hr_mean[trusted],
                    color=col_lr, lw=1.2, label="LR (ERA5 <= Nyq)",
                    zorder=ZORDER_LR,
                )
            else:
                ax.plot(
                    lam[trusted], lr_mean[trusted],
                    color=col_lr, lw=1.2, label="LR (ERA5 <= Nyq)",
                    zorder=ZORDER_LR,
                )
    else:
        # no Nyquist info → single line
        ax.plot(
            lam, lr_mean,
            color=col_lr, lw=1.2, label="LR (ERA5)",
            zorder=ZORDER_LR,
        )

    # # --- Add HR-LR crossing line and annotation ---
    # if cross_info is not None:
    #     lam_cross, k_cross = cross_info
    #     ax.axvline(x=lam_cross, color="magenta", lw=0.7, ls=":", label="HR-LR intersextion")
    #     y_max = ax.get_ylim()[1]
    #     ax.text(
    #         lam_cross,
    #         y_max * 0.45,
    #         f"k={k_cross:.3e}\nλ={lam_cross:.0f} km",
    #         color="magenta",
    #         ha="right",
    #         va="center",
    #         fontsize=7,
    #         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.55),
    #         rotation=90,
    #     )


    # --- mark low-k and high-k limits ---
    lam_low = 1.0 / low_k_max
    lam_high = 1.0 / high_k_min

    ax.axvline(lam_low, color="gray", linestyle="--",
            linewidth=0.8, alpha=0.7, zorder=ZORDER_ANNOT)

    ax.axvline(lam_high, color="gray", linestyle="--",
            linewidth=0.8, alpha=0.7, zorder=ZORDER_ANNOT)

    # Place text with x in *data* coords and y in *axes* coords (stable under log scaling / ylim changes)
    xform = ax.get_xaxis_transform()  # x=data, y=axes

    ax.text(
        lam_low * 1.029, 0.74,
        f"low-k λ={lam_low:.0f} km",
        transform=xform,
        rotation=90, color="gray", fontsize=7,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55),
        zorder=ZORDER_ANNOT,
        clip_on=False,
    )

    ax.text(
        lam_high * 1.029, 0.74,
        f"high-k λ={lam_high:.0f} km",
        transform=xform,
        rotation=90, color="gray", fontsize=7,
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55),
        zorder=ZORDER_ANNOT,
        clip_on=False,
    )


    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength λ (km)")
    ax.set_ylabel("Spectral power")
    ax.set_title("Isotropic Power Spectral Density (PSD)")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # Leave room for an external legend (keeps curves unobstructed)
    fig.subplots_adjust(right=0.76, top=0.92)
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )
    # Make legend a bit lighter
    try:
        leg.get_frame().set_alpha(0.9)
    except Exception:
        pass

    # Set y-limits after plotting (and before placing data-anchored annotations)
    ymin, ymax = ax.get_ylim()
    # Focus on dynamically relevant scales; suppress visually irrelevant LR tail
    # ax.set_ylim(bottom=max(5e-6, ymin), top=ymax)
    ax.set_ylim(bottom=1e-3, top=ymax)

    # --- annotate band ratios (more precision) ---
    lines = [
        f"low\u2011k (k ≤ {low_k_max:.3e}, λ ≥ {1.0/low_k_max:.0f} km)",
        f"  GEN / HR = {gen_hr_low_ratio:.6f}",
    ]
    if np.isfinite(lr_hr_low_ratio):
        lines.append(f"  LR  / HR = {lr_hr_low_ratio:.6f}")
    lines += [
        "",  # blank line
        f"high\u2011k (k ≥ {high_k_min:.3e}, λ ≤ {1.0/high_k_min:.0f} km)",
        f"  GEN / HR = {gen_hr_high_ratio:.6f}",
    ]
    if np.isfinite(lr_hr_high_ratio):
        lines.append(f"  LR  / HR = {lr_hr_high_ratio:.6f}")

    # --- slopes for HR, LR, PMM, and GEN (ensemble mean if available) ---
    def _get_slope(series: str, rng: str) -> float | None:
        vals = [r[2] for r in slope_rows if r[0] == series and r[1] == rng]
        return vals[0] if vals else None

    lines += ["", "slopes (log10 k – log10 P):"]
    sl_hr_hi = _get_slope("HR", "high-k")
    if sl_hr_hi is not None: lines += [f"  HR   high-k: {sl_hr_hi:.2f}"]
    sl_lr_hi = _get_slope("LR", "high-k")
    if sl_lr_hi is not None: lines += [f"  LR   high-k: {sl_lr_hi:.2f}"]
    sl_gen_ens_hi = _get_slope("GEN_ens", "high-k")
    if sl_gen_ens_hi is not None:
        lines += [f"  GEN  high-k: {sl_gen_ens_hi:.2f}"]   # GEN = ensemble mean
    sl_gen_pmm_hi = _get_slope("GEN", "high-k")
    if sl_gen_pmm_hi is not None:
        lines += [f"  PMM  high-k: {sl_gen_pmm_hi:.2f}"]
    ax.text(
        0.01,
        0.01,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82),
    )

    # ---- Baseline overlays (PSD) ----
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            dirs = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_scale_psd] resolve_baseline_dirs failed: {e}")
            dirs = {}
        labels = bo.get("labels", {})
        styles = bo.get("styles", {})
        for t, d in dirs.items():
            b_npz = d / "scale_psd_curves.npz"
            if not b_npz.exists():
                continue
            try:
                with np.load(b_npz) as bdat:
                    bk = bdat["k"]
                    # prefer generated curve if present (baseline “forecast”)
                    if "psd_gen" in bdat.files:
                        bP = bdat["psd_gen"].mean(axis=0)
                    elif "psd_lr" in bdat.files:
                        bP = bdat["psd_lr"].mean(axis=0)
                    elif "psd_lr_hr" in bdat.files:
                        bP = bdat["psd_lr_hr"].mean(axis=0)
                    else:
                        continue
                # convert to wavelength and sort like the main plot
                mask_pos_b = bk > 0.0
                lam_b = 1.0 / bk[mask_pos_b]
                ord_b = np.argsort(lam_b)[::-1]
                lam_b = lam_b[ord_b]
                bP = np.maximum(bP[mask_pos_b][ord_b], eps)
                label = labels.get(t, t)
                style = dict(styles.get(t, {}))
                style = dict(style)
                style.setdefault("zorder", ZORDER_BASELINE)
                ax.plot(lam_b, bP, label=label, **style)
            except Exception as e:
                logger.warning(f"[plot_scale_psd] Failed to overlay baseline '{t}': {e}")
        # ensure legend includes baselines
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels,
                loc="upper right", bbox_to_anchor=(0.98, 0.98),
                fontsize=8, frameon=True)
        # # never go below 1e-8 on y
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim(bottom=5e-6, top=max(ymax, 1e-7))        

    _savefig(fig, figs / "scale_psd.png", dpi=SET_DPI)





# ================================================================================
# 2. PSD low/high band ratio diag plot
# ================================================================================

# in sbgm/evaluate/evaluate_prcp/eval_scale/plot_scale.py

def plot_psd_lowhigh_diag(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Summarize PSD low/high band ratios across *all* days.

    Reads scale_psd_summary.csv and builds horizontal boxplots for:
      - GEN / HR (low-k)
      - GEN / LR (low-k)  [optional]
      - GEN / HR (high-k)

    Robustness:
      * Skip ratios when denominator band power is too small (near-zero).
      * Auto-switch to log x-scale when ratios span orders of magnitude.
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    csv_path = tables / "scale_psd_summary.csv"
    if not csv_path.exists():
        logger.warning(f"[plot_psd_lowhigh_diag] Did not find {csv_path} - skipping.")
        return

    # Treat denominators below this as "too small -> ratio not meaningful"
    denom_min = 1e-12
    if eval_cfg is not None:
        # optional hook if you want to expose it in config later
        denom_min = float(getattr(eval_cfg, "psd_ratio_denom_min", denom_min))

    gen_low_hr: list[float] = []
    gen_low_lr: list[float] = []
    gen_high_hr: list[float] = []

    def _to_float(s: str) -> float | None:
        s = s.strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    with open(csv_path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue

            hr_lowk   = _to_float(parts[1])
            gen_lowk  = _to_float(parts[2])
            gl_hr_csv = _to_float(parts[3])
            gl_lr_csv = _to_float(parts[4])
            hr_highk  = _to_float(parts[5])
            gen_highk = _to_float(parts[6])
            gh_hr_csv = _to_float(parts[7])

            # Prefer recomputation ONLY if denom is safely > denom_min
            gl_hr = None
            if (hr_lowk is not None and gen_lowk is not None and
                np.isfinite(hr_lowk) and np.isfinite(gen_lowk) and
                hr_lowk > denom_min and gen_lowk >= 0.0):
                gl_hr = float(gen_lowk / hr_lowk)
            elif gl_hr_csv is not None and np.isfinite(gl_hr_csv):
                gl_hr = float(gl_hr_csv)

            gl_lr = None
            if gl_lr_csv is not None and np.isfinite(gl_lr_csv) and gl_lr_csv > 0.0:
                gl_lr = float(gl_lr_csv)

            gh_hr = None
            if (hr_highk is not None and gen_highk is not None and
                np.isfinite(hr_highk) and np.isfinite(gen_highk) and
                hr_highk > denom_min and gen_highk >= 0.0):
                gh_hr = float(gen_highk / hr_highk)
            elif gh_hr_csv is not None and np.isfinite(gh_hr_csv):
                gh_hr = float(gh_hr_csv)

            # keep only positive, finite ratios
            if gl_hr is not None and np.isfinite(gl_hr) and gl_hr > 0:
                gen_low_hr.append(gl_hr)
            if gl_lr is not None and np.isfinite(gl_lr) and gl_lr > 0:
                gen_low_lr.append(gl_lr)
            if gh_hr is not None and np.isfinite(gh_hr) and gh_hr > 0:
                gen_high_hr.append(gh_hr)

    if not gen_low_hr and not gen_low_lr and not gen_high_hr:
        logger.warning("[plot_psd_lowhigh_diag] No valid ratios found - skipping plot.")
        return

    labels: list[str] = []
    data: list[np.ndarray] = []

    if gen_low_hr:
        labels.append("GEN / HR (low-k)")
        data.append(np.asarray(gen_low_hr, dtype=float))
    if gen_low_lr:
        labels.append("GEN / LR (low-k)")
        data.append(np.asarray(gen_low_lr, dtype=float))
    if gen_high_hr:
        labels.append("GEN / HR (high-k)")
        data.append(np.asarray(gen_high_hr, dtype=float))

    all_vals = np.concatenate([d[np.isfinite(d) & (d > 0)] for d in data])
    if all_vals.size == 0:
        logger.warning("[plot_psd_lowhigh_diag] No positive finite ratios - skipping.")
        return

    vmin = float(np.nanpercentile(all_vals, 1))
    vmax = float(np.nanpercentile(all_vals, 99))
    use_log = (vmax / max(vmin, 1e-12)) >= 50.0

    _nice()
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    fig.subplots_adjust(left=0.40, top=0.84)  # room for long y labels + subtitle

    bp = ax.boxplot(
        data,
        vert=False,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )
    for patch in bp.get("boxes", []):
        if hasattr(patch, "set_facecolor"):
            patch.set_facecolor("0.9") # type: ignore

    # reference line at perfect ratio
    ax.axvline(1.0, color="0.3", lw=0.8, ls="--", alpha=0.8)

    # mean annotations
    for y, arr in enumerate(data, start=1):
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            continue
        mu = float(arr.mean())
        ax.text(
            0.985, y, f"μ={mu:.2f}",
            transform=ax.get_yaxis_transform(),   # x in axes fraction, y in data coords
            ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            zorder=100,
        )

    if use_log:
        ax.set_xscale("log")
        ax.set_xlim(max(1e-3, vmin * 0.8), vmax * 1.25)
    else:
        ax.set_xscale("linear")
        ax.set_xlim(0.0, max(2.0, vmax * 1.15))

    low_k_max = float(getattr(eval_cfg, "low_k_max", 1.0/200.0)) if eval_cfg is not None else 1.0/200.0
    high_k_min = float(getattr(eval_cfg, "high_k_min", 1.0/20.0)) if eval_cfg is not None else 1.0/20.0

    ax.set_xlabel("Ratio")
    ax.set_title("PSD band ratios (all days)", pad=14)
    # fig.suptitle("PSD band ratios (all days)", fontsize=14, y=0.97)
    fig.text(
        0.2, 0.9,
        f"low-k: k ≤ {low_k_max:.3e} (λ ≥ {1.0/low_k_max:.0f} km)   |   "
        f"high-k: k ≥ {high_k_min:.3e} (λ ≤ {1.0/high_k_min:.0f} km)",
        ha="left", va="bottom", fontsize=10,
    )
    fig.subplots_adjust(left=0.30, right=0.96, bottom=0.18, top=0.80)

    all_vals = np.concatenate([np.asarray(a, float) for a in data])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if all_vals.size and (all_vals.max() / max(all_vals.min(), 1e-12)) > 100:
        ax.set_xscale("log")
    ax.grid(True, which="both", axis="x", ls=":", alpha=0.5)

    _savefig(fig, figs / "scale_psd_lowhigh.png", dpi=SET_DPI)




# ================================================================================
# 3. FSS vs scale curves
# ================================================================================

def plot_fss_curves(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    Read FSS outputs and make **one** multi-panel figure:
      - 1 subplot per base threshold (gen + LR if present)
      - 1 final subplot with all thresholds together
    Layout: 2 rows x 3 columns (covers up to 5 thresholds + 1 overview).
    If there are more than 5 thresholds, extra ones are added to the last row.
    
    Primary source:  <scale_root>/tables/scale_fss_summary.csv
    Optional source: <scale_root>/tables/scale_fss_daily.csv  (used ONLY to recover
    LR baselines, because the summary file currently does not contain LR rows.)

    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_fss_summary.csv"
    if not summary_path.exists():
        logger.warning(f"[plot_fss_curves] Did not find {summary_path} - skipping FSS plot.")
        return

    ens_summary_path = tables / "scale_fss_ens_summary.csv"
    by_thr_ens: dict[str, list[tuple[float, float]]] = {}
    if ens_summary_path.exists():
        with open(ens_summary_path, "r") as f:
            lines2 = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines2) > 1:
            header2 = lines2[0].split(",")
            rows2 = [l.split(",") for l in lines2[1:]]
            fss_cols2 = [(i, col) for i, col in enumerate(header2) if col.lower().startswith("fss_")]
            for r in rows2:
                base_thr = r[0].strip()
                if base_thr == "":
                    continue
                for idx, col in fss_cols2:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    by_thr_ens.setdefault(base_thr, []).append((scale_km, float(v)))

    # ------------------------------------------------------------
    # 1) Read summary (always available) - gives us GEN per threshold
    # ------------------------------------------------------------
    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_fss_curves] scale_fss_summary.csv is empty.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    # find FSS columns
    fss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("fss_")]
    if not fss_cols:
        logger.warning("[plot_fss_curves] No FSS_* columns found.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.5),...], "lr": [(5,0.4), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}

    for r in rows:
        base_thr = r[0].strip()  # e.g. "1.00"
        if base_thr == "":
            continue
        
        if base_thr not in by_thr:
            by_thr[base_thr] = {"gen": [], "lr": []}

        for idx, col in fss_cols:
            # col looks like "fss_5km"
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[base_thr]["gen"].append((scale_km, float(val)))

    # ------------------------------------------------------------
    # 2) Try to recover LR baselines from the *daily* CSV
    #    (evaluate_scale.py currently only writes LR lines there)
    # ------------------------------------------------------------
    daily_path = tables / "scale_fss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")  # ["date", "thr_mm", "fss_5km", ...]
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_fss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("fss_")]
            # tmp: (base_thr -> scale_km -> list of values)
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                thr_str = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                if not thr_str.endswith("_LR"):
                    continue  # only interested in LR here
                base_thr = thr_str.replace("_LR", "")
                for idx, col in d_fss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    val = r[idx].strip()
                    if val == "":
                        continue
                    v = float(val)
                    lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(v)

            # turn accumulators into means, append to by_thr
            for base_thr, per_scale in lr_acc.items():
                if base_thr not in by_thr:
                    by_thr[base_thr] = {"gen": [], "lr": []}
                for scale_km, vals in per_scale.items():
                    m = float(np.mean(vals))
                    by_thr[base_thr]["lr"].append((scale_km, m))
    else:
        logger.debug("[plot_fss_curves] No scale_fss_daily.csv - LR baselines will not be shown.")

    # ------------------------------------------------------------
    # 3) Build figure with subplots
    # ------------------------------------------------------------
    # Sort thresholds numerically if possible
    def _thr_key(x: str) -> float:
        try:
            return float(x)
        except Exception:
            return 9e9

    thr_list = sorted(by_thr.keys(), key=_thr_key)
    n_thr = len(thr_list)
    n_panels = n_thr + 1  # last panel = all thresholds

    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    _nice()
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols * 0.6, 4 * nrows * 0.65), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    # color cycle for per-threshold lines (will re-use in the last panel)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # Baseline overlay config (optional)
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            from sbgm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs
            _baseline_dirs_fss = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_fss_curves] resolve_baseline_dirs failed: {e}")
            _baseline_dirs_fss = {}
        _baseline_labels = bo.get("labels", {})
        _baseline_styles = bo.get("styles", {})
    else:
        _baseline_dirs_fss = {}
        _baseline_labels = {}
        _baseline_styles = {}

    # -- 3a) individual panels --
    for i, thr in enumerate(thr_list):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]
        data = by_thr[thr]
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []
        # color chosen per threshold
        color_thr = colors[i % len(colors)]

        # Ensemble mean (if available): solid colored line
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        if ens_pairs:
            ax.plot([p[0] for p in ens_pairs], [p[1] for p in ens_pairs],
                    linestyle="-", linewidth=1.8, marker="o", markersize=3.5,
                    color=COL_ENS, label="Gen (ens)")

        # PMM (formerly "gen"): same color, dashed
        gen_pairs = sorted(data["gen"], key=lambda t: t[0])
        if gen_pairs:
            ax.plot([p[0] for p in gen_pairs], [p[1] for p in gen_pairs],
                    linestyle="-.", linewidth=1.4, marker=".", color=COL_PMM, markersize=2.5,
                    label="PMM")

        # LR baseline (grey), if we reconstructed it
        lr_pairs = sorted(data["lr"], key=lambda t: t[0]) if data["lr"] else []
        if lr_pairs:
            ax.plot([p[0] for p in lr_pairs], [p[1] for p in lr_pairs],
                    marker="x", linestyle="--", linewidth=1.0, color=COL_LR, markersize=2,
                    label="LR")

        # ---- Baseline overlays: per-threshold panel only ----
        if _baseline_dirs_fss:
            for t, d in _baseline_dirs_fss.items():
                sp_b = d / "scale_fss_summary.csv"
                if not sp_b.exists():
                    continue
                try:
                    with open(sp_b, "r") as fb:
                        lines_b = [l.strip() for l in fb.readlines() if l.strip()]
                    if not lines_b:
                        continue
                    header_b = lines_b[0].split(",")
                    rows_b = [l.split(",") for l in lines_b[1:]]
                    fss_cols_b = [(idx, col) for idx, col in enumerate(header_b) if col.lower().startswith("fss_")]
                    # collect points for the current threshold only
                    pts = []
                    for r in rows_b:
                        base_thr_b = r[0].strip()
                        if base_thr_b != thr:
                            continue
                        for idx, col in fss_cols_b:
                            try:
                                scale_km = float(col.split("_")[1].replace("km", ""))
                            except Exception:
                                continue
                            v = r[idx].strip()
                            if v == "":
                                continue
                            pts.append((scale_km, float(v)))
                    if not pts:
                        continue
                    pts = sorted(pts, key=lambda p: p[0])
                    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                    label = _baseline_labels.get(t, t)
                    style = dict(_baseline_styles.get(t, {}))
                    ax.plot(xs, ys, label=label, **style)
                except Exception as e:
                    logger.warning(f"[plot_fss_curves] Failed to overlay FSS baseline '{t}': {e}")

        # Only set labels on leftmost and bottom plots
        if col == 0:
            ax.set_ylabel("FSS")
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"FSS vs scale. Thr ≥ {float(thr):.0f} mm/day")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, ls=":", alpha=0.4)

     # -- 3b) combined panel (ensemble-only, one line per threshold) --
    last_row = (n_panels - 1) // ncols
    last_col = (n_panels - 1) % ncols
    ax_all = axs[last_row, last_col]
    ax_all.set_title("FSS vs scale (ensemble only)")

    any_ens = False
    for i, thr in enumerate(thr_list):
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        if not ens_pairs:
            continue
        xs = [p[0] for p in ens_pairs]
        ys = [p[1] for p in ens_pairs]
        # one line per threshold (ensemble mean only)
        ax_all.plot(
            xs, ys,
            linewidth=1.6,
            marker=".",
            label=f"≥ {float(thr):.0f} mm"
        )
        any_ens = True

    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("FSS")
    ax_all.set_ylim(0.0, 1.0)
    ax_all.grid(True, ls=":", alpha=0.4)

    # compact legend outside the axis
    if any_ens:
        h, l = ax_all.get_legend_handles_labels()
        ax_all.legend(h, l, ncol=1, fontsize=8, frameon=True,
                      loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        ax_all.text(0.5, 0.5, "No ensemble summary available",
                    ha="center", va="center", transform=ax_all.transAxes, fontsize=9, alpha=0.7)

    # turn off any unused axes (if grid has extra cells)
    for j in range(n_panels, nrows * ncols):
        r, c = divmod(j, ncols)
        axs[r, c].axis("off")

    _savefig(fig, figs / "scale_fss.png", dpi=SET_DPI)




# ================================================================================
# 4. ISS at scales
# ================================================================================

def plot_iss_curves(scale_root: Path, eval_cfg: Any | None = None) -> None:
    """
    ISS vs scale, same layout as FSS:
      - 1 panel per threshold (PMM + Gen(ens) if available + LR if present)
      - 1 final panel with all thresholds together

    Reads:
      - <scale_root>/tables/scale_iss_summary.csv        (always)
      - <scale_root>/tables/scale_iss_daily.csv          (to recover LR baselines)
      - <scale_root>/tables/scale_iss_ens_summary.csv    (optional ensemble-mean)

    NOTE:
      The daily CSV encodes LR baselines by suffixing the threshold, e.g. "1.00_LR".
      We therefore normalize thresholds consistently (two decimals) across files.
    """
    tables = scale_root / "tables"
    figs = _ensure_dir(scale_root / "figures")
    summary_path = tables / "scale_iss_summary.csv"

    def _norm_thr(s: str) -> str:
        s = s.strip()
        if s == "":
            return ""
        # allow tags like "1.00_LR"
        if s.endswith("_LR"):
            s = s[:-3]
        try:
            return f"{float(s):.2f}"
        except Exception:
            # last resort: return as-is
            return s

    # --- read ensemble ISS summary (optional) ---
    ens_summary_path = tables / "scale_iss_ens_summary.csv"
    by_thr_ens: dict[str, list[tuple[float, float]]] = {}
    if ens_summary_path.exists():
        with open(ens_summary_path, "r") as f:
            lines2 = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines2) > 1:
            header2 = lines2[0].split(",")
            rows2 = [l.split(",") for l in lines2[1:]]
            iss_cols2 = [(i, col) for i, col in enumerate(header2) if col.lower().startswith("iss_")]
            for r in rows2:
                base_thr_raw = r[0].strip()
                base_thr = _norm_thr(base_thr_raw)
                if base_thr == "":
                    continue
                for idx, col in iss_cols2:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip()
                    if v == "":
                        continue
                    by_thr_ens.setdefault(base_thr, []).append((scale_km, float(v)))
    else:
        logger.debug("[plot_iss_curves] No scale_iss_ens_summary.csv - ensemble means will not be shown.")

    if not summary_path.exists():
        logger.warning(f"[plot_iss_curves] Did not find {summary_path} - skipping ISS plot.")
        return

    with open(summary_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if not lines:
        logger.warning("[plot_iss_curves] scale_iss_summary.csv is empty - skipping.")
        return

    header = lines[0].split(",")
    rows = [l.split(",") for l in lines[1:]]

    iss_cols = [(i, col) for i, col in enumerate(header) if col.lower().startswith("iss_")]
    if not iss_cols:
        logger.warning("[plot_iss_curves] No ISS_* columns found - skipping.")
        return

    # by_thr["1.00"] = {"gen": [(5,0.7), ...], "lr": [(5,0.8), ...]}
    by_thr: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for r in rows:
        thr_raw = r[0].strip()
        thr_id = _norm_thr(thr_raw)
        if not thr_id:
            continue
        by_thr.setdefault(thr_id, {"gen": [], "lr": []})
        for idx, col in iss_cols:
            try:
                scale_km = float(col.split("_")[1].replace("km", ""))
            except Exception:
                continue
            val = r[idx].strip()
            if val == "":
                continue
            by_thr[thr_id]["gen"].append((scale_km, float(val)))

    # try to get LR baselines from daily CSV
    daily_path = tables / "scale_iss_daily.csv"
    if daily_path.exists():
        with open(daily_path, "r") as f:
            d_lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(d_lines) > 1:
            d_header = d_lines[0].split(",")
            d_rows = [l.split(",") for l in d_lines[1:]]
            d_iss_cols = [(i, col) for i, col in enumerate(d_header) if col.lower().startswith("iss_")]
            lr_acc: dict[str, dict[float, list[float]]] = {}
            for r in d_rows:
                # daily file schema: date, thr_mm, iss_5km, ...
                if len(r) < 2:
                    continue
                thr_raw = r[1].strip()  # e.g. "1.00" or "1.00_LR"
                is_lr = thr_raw.endswith("_LR")
                if not is_lr:
                    continue
                base_thr = _norm_thr(thr_raw)
                if base_thr not in by_thr:
                    # Still allow it: create the threshold bucket if it wasn't in summary
                    by_thr.setdefault(base_thr, {"gen": [], "lr": []})
                for idx, col in d_iss_cols:
                    try:
                        scale_km = float(col.split("_")[1].replace("km", ""))
                    except Exception:
                        continue
                    v = r[idx].strip() if idx < len(r) else ""
                    if v == "":
                        continue
                    try:
                        lr_acc.setdefault(base_thr, {}).setdefault(scale_km, []).append(float(v))
                    except Exception:
                        continue
            # push averaged LR back into by_thr
            for thr, scales in lr_acc.items():
                for scale_km, arr in scales.items():
                    if len(arr) == 0:
                        continue
                    mean_lr = float(np.mean(arr))
                    by_thr[thr]["lr"].append((scale_km, mean_lr))
    else:
        logger.debug("[plot_iss_curves] No scale_iss_daily.csv - LR baselines will not be shown.")

    # sort thresholds numerically (robust to formatting)
    def _thr_key(s: str) -> float:
        try:
            return float(s)
        except Exception:
            return 9e9

    thrs_sorted = sorted(by_thr.keys(), key=_thr_key)
    n_thr = len(thrs_sorted)
    n_panels = n_thr + 1  # overview
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))

    _nice()
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols * 0.6, 4 * nrows * 0.65),
        sharex=True,
        sharey=True,
    )
    axs = np.atleast_2d(axs)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])

    # Baseline overlay config (optional)
    bo = getattr(eval_cfg, "baselines_overlay", None) if eval_cfg is not None else None
    if bo and bo.get("enabled", False):
        try:
            from sbgm.evaluate.evaluate_prcp.overlay_utils import resolve_baseline_dirs
            _baseline_dirs_iss = resolve_baseline_dirs(
                sample_root=bo["sample_root"],
                types=tuple(bo.get("types", ())),
                split=str(bo.get("split", "test")),
                eval_type="scale",
            )
        except Exception as e:
            logger.warning(f"[plot_iss_curves] resolve_baseline_dirs failed: {e}")
            _baseline_dirs_iss = {}
        _baseline_labels = bo.get("labels", {})
        _baseline_styles = bo.get("styles", {})
    else:
        _baseline_dirs_iss = {}
        _baseline_labels = {}
        _baseline_styles = {}

    for i, thr in enumerate(thrs_sorted):
        row = i // ncols
        col = i % ncols
        ax = axs[row, col]

        # ensemble mean curve (if present)
        ens_pairs = sorted(by_thr_ens.get(thr, []), key=lambda t: t[0])
        if ens_pairs:
            ax.plot(
                [p[0] for p in ens_pairs],
                [p[1] for p in ens_pairs],
                marker=".",
                linewidth=1.4,
                color=COL_ENS,
                label="Gen (ens)",
            )

        # PMM curve
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        if gen_pts:
            ax.plot(
                [p[0] for p in gen_pts],
                [p[1] for p in gen_pts],
                marker=".",
                linewidth=1.4,
                color=COL_PMM,
                label="PMM",
            )

        # LR baseline curve (if present)
        lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        if lr_pts:
            ax.plot(
                [p[0] for p in lr_pts],
                [p[1] for p in lr_pts],
                linestyle="--",
                marker="x",
                linewidth=1.0,
                color=COL_LR,
                label="LR",
            )

        # ---- Baseline overlays: per-threshold panel only ----
        if _baseline_dirs_iss:
            for t, d in _baseline_dirs_iss.items():
                sp_b = d / "scale_iss_summary.csv"
                if not sp_b.exists():
                    continue
                try:
                    with open(sp_b, "r") as fb:
                        lines_b = [l.strip() for l in fb.readlines() if l.strip()]
                    if not lines_b:
                        continue
                    header_b = lines_b[0].split(",")
                    rows_b = [l.split(",") for l in lines_b[1:]]
                    iss_cols_b = [(idx, col) for idx, col in enumerate(header_b) if col.lower().startswith("iss_")]
                    pts = []
                    for rr in rows_b:
                        thr_b = _norm_thr(rr[0].strip())
                        if thr_b != thr:
                            continue
                        for idx, colname in iss_cols_b:
                            try:
                                scale_km = float(colname.split("_")[1].replace("km", ""))
                            except Exception:
                                continue
                            v = rr[idx].strip() if idx < len(rr) else ""
                            if v == "":
                                continue
                            pts.append((scale_km, float(v)))
                    if not pts:
                        continue
                    pts = sorted(pts, key=lambda p: p[0])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    label = _baseline_labels.get(t, t)
                    style = dict(_baseline_styles.get(t, {}))
                    ax.plot(xs, ys, label=label, **style)
                except Exception as e:
                    logger.warning(f"[plot_iss_curves] Failed to overlay ISS baseline '{t}': {e}")

        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"ISS vs scale. Thr ≥ {float(thr):.0f} mm")
        
        if row == nrows - 1:
            ax.set_xlabel("Neighborhood scale (km)")
        if col == 0:
            ax.set_ylabel("ISS vs scale.")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=9, loc="lower right")

    # overview panel in the last slot
    last_idx = n_panels - 1
    ov_row = last_idx // ncols
    ov_col = last_idx % ncols
    ax_all = axs[ov_row, ov_col]

    for i, thr in enumerate(thrs_sorted):
        gen_pts = sorted(by_thr[thr]["gen"], key=lambda p: p[0])
        if not gen_pts:
            continue
        xs = [p[0] for p in gen_pts]
        ys = [p[1] for p in gen_pts]
        colr = colors[i % len(colors)]
        ax_all.plot(xs, ys, marker=".", linewidth=1.2, color=colr)
        ax_all.text(xs[-1] * 1.01, ys[-1], f"≥ {float(thr):.0f} mm", color=colr, fontsize=4, va="center")

        # lr_pts = sorted(by_thr[thr]["lr"], key=lambda p: p[0]) if by_thr[thr]["lr"] else []
        # if lr_pts:
        #     ax_all.plot([p[0] for p in lr_pts], [p[1] for p in lr_pts],
        #                 linestyle="--", linewidth=0.8, color="0.5", alpha=0.7)

    ax_all.set_ylim(0.3, 1.01)
    ax_all.set_xlabel("Neighborhood scale (km)")
    ax_all.set_ylabel("ISS")
    ax_all.set_title("ISS vs scale (all thresholds)")
    ax_all.grid(True, ls=":", alpha=0.5)

    # turn off empty axes (if any)
    for j in range(n_panels, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axs[r, c].axis("off")

    _savefig(fig, figs / "scale_iss_curves.png", dpi=SET_DPI)



# ================================================================================
# Master entry point
# ================================================================================

def plot_scale(eval_root: str | Path, eval_cfg: Any | None = None) -> None:
    """
    Master entry point - call this from evaluate_scale.py

    baseline_eval_dirs is kept here for symmetry with your old plotting module,
    but we don’t actually use it yet (easy to add later).
    """
    scale_root = Path(eval_root)
    if not scale_root.exists():
        logger.warning(f"[plot_scale] {scale_root} does not exist.")
        return

    plot_scale_psd(scale_root, eval_cfg=eval_cfg)
    plot_fss_curves(scale_root, eval_cfg=eval_cfg)
    plot_iss_curves(scale_root, eval_cfg=eval_cfg)
    plot_psd_lowhigh_diag(scale_root, eval_cfg=eval_cfg)
    logger.info(f"[plot_scale] Plots written to {scale_root / 'figures'}")