# baselines/evaluate_baselines/evaluation_baselines.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import logging

from baselines.evaluate_baselines.eval_dataresolver_baselines import BaselineDataResolver
from scor_dm.evaluate.evaluation import EvaluationConfig  # reuse your main config class

# Reuse your existing eval blocks (same signatures as in EvaluationRunner)
from scor_dm.evaluate.evaluate_prcp.eval_distributions.evaluate_distributions import run_distributional
from scor_dm.evaluate.evaluate_prcp.eval_scale.evaluate_scale import run_scale
from scor_dm.evaluate.evaluate_prcp.eval_spatial.evaluate_spatial import run_spatial
from scor_dm.evaluate.evaluate_prcp.eval_extremes.evaluate_extremes import run_extremes
from scor_dm.evaluate.evaluate_prcp.eval_temporal.evaluate_temporal import run_temporal


logger = logging.getLogger(__name__)

# Blocks excluded for baselines (no ensembles / no sigma*)
EXCLUDED_TASKS = {
    "prcp_probabilistic",
    "prcp_sigma_control",
}

DEFAULT_TASKS = [
    "prcp_distributions",
    "prcp_scale",
    "prcp_spatial",
    "prcp_extremes",
    "prcp_temporal",
    "prcp_climatology",
]

def _norm_tasks(tasks: List[str] | None) -> List[str]:
    if not tasks:
        tasks = DEFAULT_TASKS
    normed = []
    for t in tasks:
        t = t.strip()
        if t in EXCLUDED_TASKS:
            logger.info(f"[baseline] Skipping excluded task '{t}' (not applicable to baselines).")
            continue
        normed.append(t)
    return normed

def _build_eval_cfg(root: Path, baseline_type: str, user_eval: Dict[str, Any], split: str | None = None) -> EvaluationConfig:
    """
    Build an EvaluationConfig with baseline-appropriate defaults,
    mirroring how your full model runner constructs it.
    """
    if split:
        gen_dir = root / "generation" / "baselines" / baseline_type / split
        out_dir = root / "evaluation" / "baselines" / baseline_type / split / "prcp"
    else:
        gen_dir = root / "generation" / "baselines" / baseline_type
        out_dir = root / "evaluation" / "baselines" / baseline_type / "prcp"

    # Pull commonly used fields with sensible fallbacks
    def g(key, default):
        return user_eval.get(key, default)

    return EvaluationConfig(
        gen_dir=str(gen_dir),
        out_dir=str(out_dir),

        # key flags used across blocks
        eval_land_only=bool(g("land_only", True)),
        prefer_phys=True,               # baselines mirror EDM’s physical folders
        lr_key=str(g("lr_key", "lr")),

        # grid spacings
        hr_dx_km=float(g("hr_dx_km", g("grid_km_per_px", 2.5))),
        lr_dx_km=float(g("lr_dx_km", g("lr_grid_km_per_px", 31.0))),
        grid_km_per_px=float(g("grid_km_per_px", 2.5)),
        lr_grid_km_per_px=float(g("lr_grid_km_per_px", 31.0)),

        # ensemble flags OFF for baselines
        use_ensemble=False,
        ensemble_n_members=None,
        ensemble_member_seed=1234,
        ensemble_reduction_fallback="gen",
        ensemble_cache_members=False,
        dist_ensemble_pool_mode="pmm",  # irrelevant when use_ensemble=False

        # distributional
        dist_n_bins=int(g("pixel_dist_n_bins", 80)),
        dist_vmax_percentile=float(g("pixel_dist_vmax_percentile", 99.5)),
        dist_include_lr=bool(g("pixel_dist_include_lr", True)),
        dist_save_cap=int(g("pixel_dist_save_cap", 200_000)),

        # extremes
        ext_agg_kind=str(g("ext_agg_kind", "mean")),
        ext_rxk_days=tuple(g("ext_rxk_days", (1, 5))),
        ext_gev_rps_years=tuple(g("ext_gev_rps_years", (2, 5, 10, 20, 50))),
        ext_blocks_per_year=float(g("ext_blocks_per_year", 1.0)),
        ext_pot_thr_kind=str(g("ext_pot_thr_kind", "hr_quantile")),
        ext_pot_thr_val=float(g("ext_pot_thr_val", 0.95)),
        ext_pot_rps_years=tuple(g("ext_pot_rps_years", (2, 5, 10, 20, 50))),
        ext_days_per_year=float(g("ext_days_per_year", 365.25)),
        ext_wet_threshold_mm=float(g("ext_wet_threshold_mm", 1.0)),
        include_lr=bool(g("ext_include_lr", True)),
        ext_tails_basis=str(g("ext_tails_basis", "pooled_pixels")),

        # scale
        low_k_max=float(g("low_k_max", 1.0/200.0)),
        high_k_min=float(g("high_k_min", 1.0/20.0)),
        fss_thresholds_mm=tuple(g("fss_thresholds_mm", g("thresholds_mm", (1.0, 5.0, 10.0)))),
        fss_scales_km=tuple(g("fss_scales_km", (5, 10, 20))),
        compute_lr_fss=bool(g("compute_lr_fss", True)),

        # ISS
        iss_thresholds_mm=tuple(g("iss_thresholds_mm", g("thresholds_mm", (1.0, 5.0, 10.0)))),
        iss_scales_km=tuple(g("iss_scales_km", (5, 10, 20))),
        compute_lr_iss=bool(g("compute_lr_iss", True)),

        # spatial
        spatial_corr_kinds=tuple(g("spatial_corr_kinds", ("pearson", "spearman"))),
        spatial_deseasonalize=bool(g("spatial_deseasonalize", True)),
        spatial_vmin=(float(g("spatial_vmin", None)) if g("spatial_vmin", None) is not None else None),
        spatial_vmax=(float(g("spatial_vmax", None)) if g("spatial_vmax", None) is not None else None),
        spatial_show_diff=bool(g("spatial_show_diff", True)),
        spatial_include_gen=True,
        spatial_include_ens=False,
        spatial_include_hr=True,
        spatial_include_lr=True,

        # temporal
        temporal_include_lr=bool(g("temporal_include_lr", True)),
        temporal_wet_thr_mm=float(g("temporal_wet_thr_mm", 1.0)),
        temporal_max_lag=int(g("temporal_max_lag", 30)),
        temporal_max_spell=int(g("temporal_max_spell", 25)),
        temporal_group_by=str(g("temporal_group_by", "year")),
        temporal_ensemble_pool_mode="member_mean",

        # climatology / seasons
        seasons=tuple(g("seasons", ("ALL","DJF","MAM","JJA","SON"))),

        # plotting
        make_plots=bool(g("make_plots", True)),
    )

def run_baseline_evaluation(
    cfg: Dict[str, Any],
    *,
    baseline_type: str,
    split: str | None = None,    
    make_plots: bool = True,
) -> Path:
    """
    Evaluate a baseline method using the SAME evaluation blocks as EDM runs.
    Inputs are read from:
      generation/baselines/<baseline_type>/{lr_hr,pmm}/<YYYYMMDD>.npz
    Outputs are written to:
      evaluation/baselines/<baseline_type>/prcp/<eval_type>/(figures|tables)
    """
    root = Path(cfg["paths"]["root"])  # e.g., ".../models_and_samples/generated_samples"
    eval_section = cfg.get("eval", {})

    # Build eval config aligned with your model runner
    ev = _build_eval_cfg(root, baseline_type, eval_section, split=split)

    gen_root = Path(ev.gen_dir)               # .../generation/baselines/<baseline_type>
    eval_root = Path(ev.out_dir)              # .../evaluation/baselines/<baseline_type>/prcp
    eval_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"[baseline] gen_root:  {gen_root}")
    logger.info(f"[baseline] eval_root: {eval_root}")

    # Resolver sees both lr_hr/ and pmm/
    resolver = BaselineDataResolver(gen_root, variable="prcp")
    dates = resolver.list_dates()
    logger.info(f"[baseline] Found {len(dates)} dates for baseline '{baseline_type}'.")

    tasks = _norm_tasks(eval_section.get("tasks"))

    # Dispatch (same call signatures as EvaluationRunner)
    for task in tasks:
        logger.info(f"[baseline] Running task: {task}")
        t = task.strip().lower()

        if t in ("prcp_distributions","prcp_dist","distributional","dist"):
            out_dir = eval_root / "distributional"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_distributional(resolver=resolver, eval_cfg=ev, out_root=out_dir, plot_only=False)

        elif t in ("prcp_scale","scale","prcp_psd","scale_dependent"):
            out_dir = eval_root / "scale"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_scale(resolver=resolver, eval_cfg=ev, out_root=out_dir, plot_only=False)

        elif t in ("prcp_spatial","spatial","spatial_maps"):
            out_dir = eval_root / "spatial"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_spatial(resolver=resolver, eval_cfg=ev, out_root=out_dir)

        elif t in ("prcp_extremes","prcp_ext","extremes","ext"):
            out_dir = eval_root / "extremes"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_extremes(resolver=resolver, eval_cfg=ev, out_root=out_dir, plot_only=False)

        elif t in ("prcp_temporal","temporal","time","timeseries"):
            out_dir = eval_root / "temporal"
            out_dir.mkdir(parents=True, exist_ok=True)
            run_temporal(resolver=resolver, eval_cfg=ev, out_root=out_dir,
                         group_by=ev.temporal_group_by, seasons=ev.seasons, make_plots=ev.make_plots)

        else:
            logger.warning(f"[baseline] Unknown task '{task}' — skipping.")

    logger.info(f"[baseline] Done. Outputs at: {eval_root}")
    return eval_root


# === Helper: run all baseline types/splits from cfg (for CLI integration) ===
def run_all_baselines(cfg) -> Dict[tuple, Path]:
    """
    Convenience wrapper to mirror the old baseline_eval.run_all() behavior.
    Reads baseline types and splits from cfg and calls run_baseline_evaluation for each.

    Expected cfg structure:
      cfg["paths"]["root"]: base path to models_and_samples/generated_samples
      cfg["eval"]["tasks"]: list of tasks (optional)
      cfg["baseline"]["types"]: list like ["bilinear", "qm", "unet_sr"] (optional; default: all three)
      cfg["baseline"]["splits"]: list like ["train","valid","test"] (optional; default: ["test"])  
    """
    results: Dict[tuple, Path] = {}

    base_cfg_eval = cfg.get("eval", {})
    base_root = Path(cfg["paths"]["root"])  # raises if missing → fail fast

    bl_cfg = cfg.get("baseline", {})
    types = bl_cfg.get("types", ["bilinear", "qm", "unet_sr"])  # default trio
    splits = bl_cfg.get("splits", ["test"])                        # default test only

    for btype in types:
        for split in splits:
            logger.info(f"[baseline] === {btype} | {split} ===")
            out = run_baseline_evaluation(cfg, baseline_type=btype, split=split, make_plots=bool(base_cfg_eval.get("make_plots", True)))
            results[(btype, split)] = out
    return results