    # sbgm/evaluate/evaluate_prcp/eval_distributional/evaluate_distributional.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import numpy as np
import torch

from sbgm.evaluate.evaluate_prcp.eval_distributions.metrics_distributions import (
    collect_pooled_distributions,
    collect_daily_histograms,
    compute_distributional_metrics,
    collect_ensemble_histograms
)
from sbgm.evaluate.evaluate_prcp.eval_distributions.plot_distributions import (
    plot_pooled_distribution,
    plot_seasonal_distributions,
)


logger = logging.getLogger(__name__)

# --- Task list and helpers for distributional evaluation ---
# Symmetric with extremes: tasks describe what to COMPUTE; plotting is controlled by output_plots.
# Backwards compatibility:
#   - old names like pooled_hist/daily_hist/ensemble_hist_* still work
#   - plot_pooled/plot_seasons are treated as aliases for pooled/seasonal (compute + plot if output_plots)
SUPPORTED_TASKS = [
    "pooled",                 # pooled pixel distribution (bins + hr/gen/lr hists)
    "daily",                  # per-day histograms (CI bands)
    "ensemble_pool",          # ensemble pooled histogram
    "ensemble_member_mean",   # ensemble member-mean pdf + spread
    "metrics",                # distributional metrics table
    "seasonal",               # seasonal distributions (requires pooled bins/tables)
]

DEFAULT_TASKS_BASE = ["pooled", "daily", "metrics", "seasonal"]

_TASK_ALIASES = {
    # old compute task names
    "pooled_hist": "pooled",
    "hist_pooled": "pooled",
    "daily_hist": "daily",
    "hist_daily": "daily",
    "ensemble_hist_pool": "ensemble_pool",
    "ensemble_hist": "ensemble_pool",
    "ensemble_hist_member_mean": "ensemble_member_mean",
    "ensemble_hist_mean": "ensemble_member_mean",
    # old plotting tasks -> treat as compute tasks; plotting is handled by output_plots
    "plot": "pooled",
    "plot_pooled": "pooled",
    "pooled_plot": "pooled",
    "main_plot": "pooled",
    "plot_seasons": "seasonal",
    "seasonal_plot": "seasonal",
    "seasons_plot": "seasonal",
}

def _as_dict(obj):
    # Safely convert OmegaConf/dict/dataclass-like to dict
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "items"):
        try:
            return dict(obj.items())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass
    return {}

def _get_family_cfg(eval_cfg, keys):
    # keys: list of possible family keys (in priority order)
    fam = None
    # Try families
    families = getattr(eval_cfg, "families", None)
    if families is not None:
        for k in keys:
            if hasattr(families, k):
                fam = getattr(families, k)
                break
            if isinstance(families, dict) and k in families:
                fam = families[k]
                break
    # Try family_plans
    if fam is None:
        family_plans = getattr(eval_cfg, "family_plans", None)
        if family_plans is not None:
            for k in keys:
                if hasattr(family_plans, k):
                    fam = getattr(family_plans, k)
                    break
                if isinstance(family_plans, dict) and k in family_plans:
                    fam = family_plans[k]
                    break
    # Try direct attribute
    if fam is None:
        for k in keys:
            if hasattr(eval_cfg, k):
                fam = getattr(eval_cfg, k)
                break
            if isinstance(eval_cfg, dict) and k in eval_cfg:
                fam = eval_cfg[k]
                break
    return _as_dict(fam)

def _norm_tasks(task_list):
    if not isinstance(task_list, (list, tuple)):
        return []
    out = []
    seen = set()
    for t in task_list:
        if not isinstance(t, str):
            continue
        tl = t.strip().lower()
        tl = _TASK_ALIASES.get(tl, tl)
        if tl not in seen:
            out.append(tl)
            seen.add(tl)
    return out

def _warn_unknown_tasks(tasks):
    for t in tasks:
        if t not in SUPPORTED_TASKS:
            logger.warning(f"[prcp_distributional] Unknown task requested: '{t}' (supported: {SUPPORTED_TASKS})")

def run_distributional(
    resolver,
    eval_cfg,
    out_root: str | Path,
    *,
    plot_only: bool = False,
) -> None:
    """
    Distributional (1D, pooled-pixel) evaluation - now supports task-list driven exeution
    """
    out_root = Path(out_root)
    tables_dir = out_root / "tables"
    figs_dir = out_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve family config (prcp_distributions) ---
    fam = _get_family_cfg(eval_cfg, [
        "prcp_distributions", "prcp_distributional", "prcp_distribution", "distributional"
    ])
    enabled = fam.get("enabled", None)
    if enabled is not None and not bool(enabled):
        logger.info("[prcp_distributions] Family disabled via config - skipping.")
        return

    # Output toggles: prefer explicit family knobs; if missing, derive from task_list; else fallback to legacy
    output_plots = fam.get("output_plots", None)
    output_metrics = fam.get("output_metrics", None)

    # Output toggles (symmetric with extremes):
    #  - tasks control what to compute
    #  - output_plots controls whether we call plotting functions for whatever was requested/computed
    output_plots = fam.get("output_plots", None)
    output_metrics = fam.get("output_metrics", None)

    if output_plots is None:
        output_plots = bool(getattr(eval_cfg, "make_plots", True))
    output_plots = bool(output_plots)

    if output_metrics is None:
        output_metrics = (not plot_only)
    output_metrics = bool(output_metrics)

        # --- Task list selection ---
    selected_tasks = fam.get("task_list", None)

    if not selected_tasks or (isinstance(selected_tasks, (list, tuple)) and len(selected_tasks) == 0):
        selected_tasks = list(DEFAULT_TASKS_BASE)

        use_ensemble = bool(getattr(eval_cfg, "use_ensemble", False))
        mode = str(getattr(eval_cfg, "dist_ensemble_pool_mode", "pool")).lower()
        if use_ensemble:
            if mode == "pool":
                selected_tasks.insert(2, "ensemble_pool")
            elif mode in ("member_mean", "mean"):
                selected_tasks.insert(2, "ensemble_member_mean")
    else:
        selected_tasks = _norm_tasks(selected_tasks)

    _warn_unknown_tasks(selected_tasks)

    # Ensure prerequisites: daily/ensemble/metrics/seasonal want pooled bins when computing
    if any(t in selected_tasks for t in ("daily", "ensemble_pool", "ensemble_member_mean", "metrics", "seasonal")) and ("pooled" not in selected_tasks):
        selected_tasks = ["pooled"] + [t for t in selected_tasks if t != "pooled"]

    # --- plot_only hard override ---
    # If plot_only True, skip all compute/metrics tasks, only execute plotting if requested.
    plot_tasks = {"plot_pooled", "plot_seasons"}
    if plot_only:
        if not output_plots:
            logger.info("[prcp_distributional] plot_only=True but output_plots=False -> skipping.")
            return

        # If user requests seasonal, render seasonal; if user requests pooled (or anything else), render pooled.
        if "seasonal" in selected_tasks:
            plot_seasonal_distributions(out_root, eval_cfg=eval_cfg)
        if "pooled" in selected_tasks or any(t in selected_tasks for t in ("daily", "ensemble_pool", "ensemble_member_mean", "metrics")):
            plot_pooled_distribution(out_root, eval_cfg=eval_cfg)

        logger.info("[prcp_distributional] plot_only=True -> done.")
        return

    # --- Compute gating ---
    want_compute = output_metrics and (not plot_only)

    # --- Gather dates ---
    dates = list(resolver.list_dates())
    if not dates:
        logger.warning("[eval_distributional] No dates from resolver - nothing to do.")
        return
    logger.info(f"[eval_distributional] Running on {len(dates)} dates.")

    # knobs
    n_bins: int = int(getattr(eval_cfg, "dist_n_bins", 80))
    vmax_pct: float = float(getattr(eval_cfg, "dist_vmax_percentile", 99.5))
    include_lr: bool = bool(getattr(eval_cfg, "dist_include_lr", True))
    save_cap: int = int(getattr(eval_cfg, "dist_save_cap", 200_000))
    # --- Task execution ---
    pooled = None
    daily = None
    bins = None
    # --- Pooled histogram ---
    if "pooled" in selected_tasks and want_compute:
        pooled = collect_pooled_distributions(
            resolver=resolver,
            dates=dates,
            include_lr=include_lr,
            n_bins=n_bins,
            vmax_percentile=vmax_pct,
            save_samples_cap=save_cap,
        )
        bins = pooled["bins"]
        # Write CSVs
        np.savetxt(tables_dir / "dist_bins.csv", bins, delimiter=",", header="bin_edge", comments="")
        def _write_series(name: str, arr: np.ndarray):
            p = tables_dir / f"dist_{name}.csv"
            with open(p, "w") as f:
                f.write("bin_idx,count\n")
                for i, c in enumerate(arr.astype(int)):
                    f.write(f"{i},{int(c)}\n")
        _write_series("hr", pooled["hr_hist"])
        _write_series("gen", pooled["gen_hist"])
        if pooled.get("lr_hist") is not None:
            _write_series("lr", pooled["lr_hist"])
        # Save capped pooled sample vectors so metrics can be computed later without reloading fields
        try:
            np.savez_compressed(
                tables_dir / "dist_pooled_samples.npz",
                hr_vec=np.asarray(pooled.get("hr_vec", np.empty((0,), dtype=np.float32))),
                gen_vec=np.asarray(pooled.get("gen_vec", np.empty((0,), dtype=np.float32))),
                lr_vec=np.asarray(pooled.get("lr_vec", np.empty((0,), dtype=np.float32))) if pooled.get("lr_vec", None) is not None else np.empty((0,), dtype=np.float32),
                bins=np.asarray(pooled.get("bins", np.empty((0,), dtype=np.float32))),
            )
        except Exception as e:
            logger.warning(f"[prcp_distributions] Could not save dist_pooled_samples.npz: {e}")            
    # If not run, but needed for downstream, try to load bins
    if bins is None:
        try:
            bins_path = tables_dir / "dist_bins.csv"
            if bins_path.exists():
                bins = np.loadtxt(bins_path, delimiter=",", skiprows=1) if bins_path.read_text().startswith("bin_edge") else np.loadtxt(bins_path, delimiter=",")
        except Exception:
            bins = None

    # --- Daily histogram ---
    if "daily" in selected_tasks and want_compute:
        if bins is None:
            logger.warning("[prcp_distributions] Cannot build daily_hist - dist_bins.csv missing and pooled_hist not run.")
        else:
            try:
                daily = collect_daily_histograms(
                    resolver=resolver,
                    dates=dates,
                    include_lr=include_lr,
                    bins=bins,
                )
                np.savez_compressed(
                    tables_dir / "dist_daily.npz",
                    **daily
                )
            except Exception as e:
                logger.warning(f"[prcp_distributions] Could not build daily histograms: {e}")

    # --- Ensemble histogram(s) ---
    ens_outs = {}
    use_ensemble = bool(getattr(eval_cfg, "use_ensemble", False))
    if use_ensemble and bins is not None:
        # Only build if requested
        ens_modes_to_run = []
        if "ensemble_pool" in selected_tasks:
            ens_modes_to_run.append("pool")
        if "ensemble_member_mean" in selected_tasks:
            ens_modes_to_run.append("member_mean")
        # De-duplicate
        ens_modes_to_run = list(dict.fromkeys(ens_modes_to_run))
        for mode in ens_modes_to_run:
            try:
                ens_out = collect_ensemble_histograms(
                    resolver=resolver,
                    dates=dates,
                    bins=bins,
                    mode=mode,
                    n_members=getattr(eval_cfg, "ensemble_n_members", None),
                    seed=int(getattr(eval_cfg, "ensemble_member_seed", 1234)),
                )
                if ens_out:
                    ens_outs[mode] = ens_out
                    # Write CSVs/NPZ as in old logic
                    if mode == "pool" and "counts_pool" in ens_out:
                        p = tables_dir / "dist_gen_ens_pool.csv"
                        with open(p, "w") as f:
                            f.write("bin_idx,count\n")
                            for i, c in enumerate(ens_out["counts_pool"].astype(int)):
                                f.write(f"{i},{int(c)}\n")
                    if mode == "member_mean" and "pdf_mean" in ens_out:
                        p = tables_dir / "dist_gen_ens_mean.csv"
                        with open(p, "w") as f:
                            f.write("bin_idx,pdf\n")
                            for i, v in enumerate(ens_out["pdf_mean"].astype(float)):
                                f.write(f"{i},{float(v)}\n")
                    # Save extra arrays for optional plotting of spread
                    np.savez_compressed(
                        tables_dir / "dist_member_histograms.npz",
                        **{k: v for k, v in ens_out.items() if k in ("bins","counts_members","n_members","pdf_mean","pdf_q10","pdf_q50","pdf_q90","mode")}
                    )
            except Exception as e:
                logger.warning(f"[prcp_distributions] Ensemble histogram build failed for mode '{mode}': {e}")

    # --- Metrics ---
    if "metrics" in selected_tasks and want_compute:
        if pooled is None:
            # Prefer loading pooled sample vectors (required for KS/W1 on samples)
            pooled_npz = tables_dir / "dist_pooled_samples.npz"
            if pooled_npz.exists():
                try:
                    d = np.load(pooled_npz)
                    pooled = {
                        "bins": np.asarray(d["bins"]),
                        "hr_vec": np.asarray(d["hr_vec"]).astype(np.float32),
                        "gen_vec": np.asarray(d["gen_vec"]).astype(np.float32),
                    }
                    lr_vec = np.asarray(d["lr_vec"]).astype(np.float32) if ("lr_vec" in d) else None
                    if lr_vec is not None and lr_vec.size > 0:
                        pooled["lr_vec"] = lr_vec
                    # Also include histograms if present (optional)
                    if "hr_hist" in d: pooled["hr_hist"] = np.asarray(d["hr_hist"]).astype(np.int64)
                    if "gen_hist" in d: pooled["gen_hist"] = np.asarray(d["gen_hist"]).astype(np.int64)
                    if "lr_hist" in d: pooled["lr_hist"] = np.asarray(d["lr_hist"]).astype(np.int64)
                except Exception as e:
                    logger.warning(f"[prcp_distributions] Failed to load dist_pooled_samples.npz: {e}")
                    pooled = None

            # Last-resort: load only histograms from CSVs (NOTE: metrics may be limited)
            if pooled is None:
                try:
                    bins_path = tables_dir / "dist_bins.csv"
                    hr_path = tables_dir / "dist_hr.csv"
                    gen_path = tables_dir / "dist_gen.csv"
                    if bins_path.exists() and hr_path.exists() and gen_path.exists():
                        bins = np.loadtxt(bins_path, delimiter=",", skiprows=1) if bins_path.read_text().startswith("bin_edge") else np.loadtxt(bins_path, delimiter=",")
                        hr_hist = np.loadtxt(hr_path, delimiter=",", skiprows=1)[:, 1]
                        gen_hist = np.loadtxt(gen_path, delimiter=",", skiprows=1)[:, 1]
                        pooled = {"bins": bins, "hr_hist": hr_hist, "gen_hist": gen_hist}
                except Exception as e:
                    logger.warning(f"[prcp_distributions] Failed to load pooled histograms from CSVs: {e}")
                    pooled = None
        # Try to load ensemble if not run
        ens_for_metrics = None
        if "ensemble_hist_pool" in ens_outs:
            ens_for_metrics = ens_outs["pool"]
        elif "ensemble_hist_member_mean" in ens_outs:
            ens_for_metrics = ens_outs["member_mean"]
        else:
            ens_for_metrics = None
        if pooled is not None:
            # compute_distributional_metrics requires pooled sample vectors for KS/W1
            if ("hr_vec" not in pooled) or ("gen_vec" not in pooled):
                logger.warning("[prcp_distributions] Skipping metrics: pooled sample vectors not available. Run pooled_hist to create dist_pooled_samples.npz.")
            else:
                try:
                    metrics_rows = compute_distributional_metrics(pooled, ensembles=ens_for_metrics)
                    with open(tables_dir / "dist_metrics.csv", "w") as f:
                        f.write("ref,comp,wasserstein,ks_stat,ks_p,kl_hr_to_x\n")
                        for r in metrics_rows:
                            f.write("{ref},{comp},{wasserstein:.6f},{ks_stat:.6f},{ks_p:.6f},{kl_hr_to_x:.6f}\n"
                                    .format(**{k: (v if v is not None else float("nan")) for k, v in r.items()}))
                except Exception as e:
                    logger.warning(f"[prcp_distributions] Could not compute/write metrics: {e}")
        else:
            logger.warning("[prcp_distributions] Skipping metrics: pooled histograms unavailable.")

    # --- Plots ---
    # Dependency checks before plotting
    def _check_plot_deps():
        missing = []
        bins_path = tables_dir / "dist_bins.csv"
        hr_path = tables_dir / "dist_hr.csv"
        gen_path = tables_dir / "dist_gen.csv"
        ens_pool_path = tables_dir / "dist_gen_ens_pool.csv"
        ens_mean_path = tables_dir / "dist_gen_ens_mean.csv"
        daily_npz = tables_dir / "dist_daily.npz"
        # Pooled plot deps
        if "plot_pooled" in selected_tasks:
            if not bins_path.exists():
                missing.append("dist_bins.csv")
            if not (hr_path.exists() or gen_path.exists()):
                missing.append("dist_hr.csv or dist_gen.csv")

        # Seasonal plot deps
        if "plot_seasons" in selected_tasks:
            if not daily_npz.exists():
                missing.append("dist_daily.npz")

        # For ensemble overlays: warn if both missing
        if ("ensemble_hist_pool" in selected_tasks or "ensemble_hist_member_mean" in selected_tasks) and (not ens_pool_path.exists() and not ens_mean_path.exists()):
            logger.warning("[prcp_distributions] Ensemble plot requested but no ensemble histogram CSVs present. Run the relevant ensemble_hist_* task.")
        return missing

    if output_plots:
        deps_missing = _check_plot_deps()
        if deps_missing:
            logger.warning(f"[prcp_distributions] Plot(s) requested but missing required files: {deps_missing}")
        else:
            try:
                if "pooled" in selected_tasks or any(t in selected_tasks for t in ("daily", "ensemble_pool", "ensemble_member_mean", "metrics")):
                    plot_pooled_distribution(out_root, eval_cfg=eval_cfg)
                if "seasonal" in selected_tasks:
                    plot_seasonal_distributions(out_root, eval_cfg=eval_cfg)
            except Exception as e:
                logger.warning(f"[prcp_distributional] Plotting failed: {e}")                