from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import torch

from sbgm.utils import get_model_string
from sbgm.evaluate.evaluation import EvaluationConfig, EvaluationRunner
from sbgm.evaluate.evaluation_summary.build_summary import build_and_write_summary

logger = logging.getLogger(__name__)


def _default_gen_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "generation" / model_str


def _default_eval_dir(cfg) -> Path:
    model_str = get_model_string(cfg)
    sample_root = cfg["paths"]["sample_dir"]
    return Path(sample_root) / "evaluation" / model_str


def _normalize_run_dir(user_dir: str | Path | None, default_dir: Path, cfg) -> Path:
    """Resolve a user-provided eval/gen directory to the model-specific run directory.

    Behavior:
      - if `user_dir` is None -> use `default_dir`
      - if `user_dir` already points to the model-specific directory -> keep it
      - otherwise treat `user_dir` as a base root and append the model string
    """
    if user_dir is None:
        return default_dir

    p = Path(user_dir)
    model_str = get_model_string(cfg)

    # Already model-specific.
    if p.name == model_str:
        return p

    # Treat as base directory and append the model-specific run folder.
    return p / model_str


def evaluation_main(cfg):
    """
        Launch modular evaluation process based on provided config.
    """
    fe = cfg.get("full_gen_eval", {})

    # --- Family key canonicalization and task list helpers ---
    def _canon_family_key(k: str) -> str:
        k2 = str(k).strip().lower()
        aliases = {
            "prcp_distributions": "prcp_distributional",
            "prcp_distribution": "prcp_distributional",
            "distributional": "prcp_distributional",
            "dist": "prcp_distributional",
        }
        return aliases.get(k2, k2)

    def _get_task_list(fam_cfg: dict) -> list[str] | None:
        if not isinstance(fam_cfg, dict):
            return None
        tl = fam_cfg.get("task_list", None)
        if tl is None:
            tl = fam_cfg.get("tasks", None)
        if tl is None:
            tl = fam_cfg.get("taks_list", None)  # tolerate common typo
        if tl is None:
            return None
        return list(tl) if isinstance(tl, (list, tuple)) else None

    def _summary_pillars_from_tasks(task_names: list[str]) -> list[str]:
        mapping = {
            "prcp_distributional": "distributional",
            "prcp_extremes": "extremes",
            "prcp_features": "features",
            "prcp_scale": "scale",
            "prcp_probabilistic": "probabilistic",
            "prcp_spatial": "climatological",
            "prcp_temporal": "temporal",
            "prcp_dates": "dates",
        }
        out: list[str] = []
        for t in task_names:
            key = mapping.get(str(t).strip().lower())
            if key is not None and key not in out:
                out.append(key)
        return out

    def _attach_summary_paths(cfg_obj: dict, gen_root_path: Path, eval_root_path: Path) -> None:
        cfg_obj["gen_root"] = str(gen_root_path)
        cfg_obj["eval_root"] = str(eval_root_path)
        cfg_obj["generated_root"] = str(gen_root_path)
        cfg_obj["evaluation_root"] = str(eval_root_path)
        paths = cfg_obj.setdefault("paths", {})
        if isinstance(paths, dict):
            paths["generation_dir"] = str(gen_root_path)
            paths["generated_samples_dir"] = str(gen_root_path)
            paths["evaluation_dir"] = str(eval_root_path)

    # standard flags
    do_prob = bool(fe.get("do_prob", True))
    do_scale = bool(fe.get("do_scale", True))
    do_ext = bool(fe.get("do_ext", True))
    do_dist = bool(fe.get("do_dist", True))   
    do_spat = bool(fe.get("do_spat", True))
    do_temp = bool(fe.get("do_temp", False))
    do_feat = bool(fe.get("do_feat", False))
    do_dates = bool(fe.get("do_dates", False))

    gen_dir = fe.get("gen_dir", None)
    eval_dir = fe.get("eval_dir", None)

    # directories
    gen_root = _normalize_run_dir(gen_dir, _default_gen_dir(cfg), cfg)
    eval_root = _normalize_run_dir(eval_dir, _default_eval_dir(cfg), cfg)
    eval_root.mkdir(parents=True, exist_ok=True)

    # seeds like old version
    seed = int(fe.get("seed", 1234))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # device is still read from your YAML
    device = torch.device(cfg["training"]["device"])

    logger.info(f"[evaluation_main] gen_root: {gen_root}")
    logger.info(f"[evaluation_main] eval_root: {eval_root}")

    # Parse baselines_overlay block if present
    bo = fe.get("baselines_overlay", {})
    baselines_overlay = {
        "enabled": bool(bo.get("enabled", False)),
        "types": tuple(bo.get("types", ())),
        "split": str(bo.get("split", "test")),
        "labels": dict(bo.get("labels", {})),
        "styles": dict(bo.get("styles", {})),
        "sample_root": str(cfg["paths"]["sample_dir"]),
    }

    # build NEW evaluation config (note: this is sbgm/evaluate/evaluation.py)
    ev_cfg = EvaluationConfig(
        gen_dir=str(gen_root),
        out_dir=str(eval_root),
        
        eval_land_only=bool(fe.get("eval_land_only", False)),
        prefer_phys=bool(fe.get("prefer_phys", True)),
        lr_key=str(fe.get("lr_key", "lr")),
        region_mask_path=fe.get("region_mask_path", None),
        grid_km_per_px=float(fe.get("grid_km_per_px", 2.5)),
        lr_grid_km_per_px=float(fe.get("lr_grid_km_per_px", 31.0)),
        seasons=tuple(fe.get("seasons", ("ALL", "DJF", "MAM", "JJA", "SON"))),
        
        hr_dx_km=float(fe.get("hr_dx_km", fe.get("grid_km_per_px", 2.5))),
        lr_dx_km=float(fe.get("lr_dx_km", fe.get("lr_grid_km_per_px", 31.0))),

        # ------ new field for baselines_overlay ------
        baselines_overlay=baselines_overlay,

        crps_examples_n_members=int(fe.get("crps_examples_n_members", 4)),

        # FSS evaluation config fields
        thresholds_mm=tuple(fe.get("thresholds_mm", (1.0, 5.0, 10.0))),
        fss_scales_km=tuple(fe.get("fss_scales_km", (5, 10, 20))),
        fss_thresholds_mm=tuple(fe.get("fss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))),
        compute_lr_fss=bool(fe.get("compute_lr_fss", True)),

        # ISS evaluation config fields  
        iss_thresholds_mm=tuple(fe.get("iss_thresholds_mm", fe.get("thresholds_mm", (1.0, 5.0, 10.0)))),
        iss_scales_km=tuple(fe.get("iss_scales_km", (5, 10, 20))),
        compute_lr_iss=bool(fe.get("compute_lr_iss", True)),
        
        # Reliability, Spread-Skill, PIT config fields
        reliability_bins=int(fe.get("reliability_bins", 10)),
        spread_skill_bins=int(fe.get("spread_skill_bins", 10)),
        pit_bins=int(fe.get("pit_bins", 20)),

        # Variogram
        variogram_p=float(fe.get("variogram_p", 0.5)),
        variogram_max_pairs=int(fe.get("variogram_max_pairs", 5000)),

        # scale-specific fields with sensible fallbacks to the old names
        low_k_max=float(fe.get("low_k_max", 1.0 / 200.0)),
        high_k_min=float(fe.get("high_k_min", 1.0 / 20.0)),
        make_plots=bool(fe.get("make_plots", True)),
        
        # Distributional evaluation config fields
        dist_n_bins=int(fe.get("pixel_dist_n_bins", 80)),
        dist_vmax_percentile=float(fe.get("pixel_dist_vmax_percentile", 99.5)),
        dist_include_lr=bool(fe.get("pixel_dist_include_lr", True)),
        dist_save_cap=int(fe.get("pixel_dist_save_cap", 200_000)),

        # Extremes evaluation config fields
        ext_agg_kind=str(fe.get("ext_agg_kind", "mean")),
        ext_rxk_days=tuple(fe.get("ext_rxk_days", (1, 5))),
        ext_gev_rps_years=tuple(fe.get("ext_gev_rps_years", (2, 5, 10, 20, 50))),
        ext_blocks_per_year=float(fe.get("ext_blocks_per_year", 1.0)),
        ext_pot_thr_kind=str(fe.get("ext_pot_thr_kind", "hr_quantile")),
        ext_pot_thr_val=float(fe.get("ext_pot_thr_val", 0.95)),
        ext_pot_rps_years=tuple(fe.get("ext_pot_rps_years", (2, 5, 10, 20, 50))),
        ext_days_per_year=float(fe.get("ext_days_per_year", 365.25)),
        ext_wet_threshold_mm=float(fe.get("ext_wet_threshold_mm", 1.0)),
        include_lr=bool(fe.get("ext_include_lr", False)),
        ext_tails_basis=str(fe.get("ext_tails_basis", "pooled_pixels")),

        # Spatial evaluation config
        spatial_corr_kinds=tuple(fe.get("spatial_corr_kinds", ("pearson", "spearman"))),
        spatial_deseasonalize=bool(fe.get("spatial_deseasonalize", True)),
        spatial_vmin=float(fe.get("spatial_vmin", None)) if fe.get("spatial_vmin", None) is not None else None,
        spatial_vmax=float(fe.get("spatial_vmax", None)) if fe.get("spatial_vmax", None) is not None else None,
        spatial_show_diff=bool(fe.get("spatial_show_diff", True)),
        spatial_include_gen=bool(fe.get("spatial_include_gen", False)),
        spatial_include_ens=bool(fe.get("spatial_include_ens", True)),
        spatial_include_hr=bool(fe.get("spatial_include_hr", True)),
        spatial_include_lr=bool(fe.get("spatial_include_lr", True)),

        # Temporal evaluation config fields
        temporal_include_lr=bool(fe.get("temporal_include_lr", True)),
        temporal_wet_thr_mm=float(fe.get("temporal_wet_thr_mm", 1.0)),
        temporal_max_lag=int(fe.get("temporal_max_lag", 30)),
        temporal_max_spell=int(fe.get("temporal_max_spell", 25)),
        temporal_group_by=str(fe.get("temporal_group_by", "year")),
        temporal_ensemble_pool_mode=str(fe.get("temporal_ensemble_pool_mode", "member_mean")),

        # Dates evaluation config fields
        dates_list = fe.get("dates_list", []),
        dates_max = int(fe.get("dates_max", 4)),
        dates_include_lr = bool(fe.get("dates_include_lr", True)),
        dates_include_members = bool(fe.get("dates_include_members", True)),
        dates_n_members = int(fe.get("dates_n_members", 3)),
        dates_cmap = str(fe.get("dates_cmap", "Blues")),
        dates_percentile = float(fe.get("dates_percentile", 99.5)),
        
        # Ensemble evaluation config fields
        use_ensemble=bool(fe.get("use_ensemble", True)),
        ensemble_n_members=fe.get("ensemble_n_members", None),
        ensemble_member_seed=int(fe.get("ensemble_member_seed", 1234)),
        ensemble_reduction_fallback=str(fe.get("ensemble_reduction_fallback", "pmm")),
        ensemble_cache_members=bool(fe.get("ensemble_cache_members", False)),
        dist_ensemble_pool_mode=str(fe.get("dist_ensemble_pool_mode", "pool")),

        # Features evaluation config fields
        sal_structure_mode=str(fe.get("sal_structure_mode", "object")),
        sal_threshold_kind=str(fe.get("sal_threshold_kind", "quantile")),  # unified singular name
        sal_threshold_value=float(fe.get("sal_threshold_value", 0.90)),
        sal_connectivity=int(fe.get("sal_connectivity", 8)),
        sal_min_area_px=int(fe.get("sal_min_area_px", 9)),
        sal_smooth_sigma=(float(_tmp) if (_tmp := fe.get("sal_smooth_sigma", 0.75)) is not None else None),
        sal_peakedness_mode=str(fe.get("sal_peakedness_mode", "largest")),

    )

    # --- Determine tasks to run ---
    families = fe.get("families", None)

    # Legacy defaults (backwards compatible)
    legacy_enabled = {
        "prcp_probabilistic": do_prob,
        "prcp_scale": do_scale,
        "prcp_extremes": do_ext,
        "prcp_distributional": do_dist,
        "prcp_spatial": do_spat,
        "prcp_temporal": do_temp,
        "prcp_features": do_feat,
        "prcp_dates": do_dates,
    }

    family_plans: dict[str, dict] = {}
    tasks: list[str] = []

    if isinstance(families, dict) and len(families) > 0:
        for k, v in families.items():
            fam_key = _canon_family_key(k)
            fam_cfg = v if isinstance(v, dict) else {}

            enabled = fam_cfg.get("enabled", None)
            if enabled is None:
                enabled = bool(legacy_enabled.get(fam_key, False))
            enabled = bool(enabled)

            # output toggles: defaults derived from global plot_only
            output_plots = fam_cfg.get("output_plots", None)
            if output_plots is None:
                output_plots = True
            output_plots = bool(output_plots)

            output_metrics = fam_cfg.get("output_metrics", None)
            if output_metrics is None:
                output_metrics = (not bool(fe.get("plot_only", False)))
            output_metrics = bool(output_metrics)

            task_list = _get_task_list(fam_cfg)

            family_plans[fam_key] = {
                "enabled": enabled,
                "output_plots": output_plots,
                "output_metrics": output_metrics,
                "task_list": task_list,
            }
            if enabled:
                tasks.append(fam_key)

        # If user provided families but none enabled, do nothing explicitly.
    else:
        # map YAML flags -> new modular task names (legacy)
        tasks = [k for k, en in legacy_enabled.items() if en]

    # Make family plans visible to per-family evaluation modules
    if family_plans:
        try:
            ev_cfg.family_plans = family_plans
        except Exception as e:
            logger.warning(f"Could not set family_plans in EvaluationConfig: {e}")
            pass

    runner = EvaluationRunner(
        cfg_yaml=cfg,
        eval_cfg=ev_cfg,
        device=device,
        baseline_eval_dirs=None,
        plot_only=bool(fe.get("plot_only", False)),
        family_plans=family_plans if family_plans else None,
    )

    runner.run(tasks=tasks)

    # Build a compact paper-oriented summary after the modular evaluation run.
    write_summary = bool(fe.get("write_summary", True))
    if write_summary:
        try:
            _attach_summary_paths(cfg, gen_root, eval_root)
            summary_pillars = _summary_pillars_from_tasks(tasks)
            if summary_pillars:
                summary_out = build_and_write_summary(
                    cfg=cfg,
                    pillars=summary_pillars,
                    out_path=eval_root / "summary" / "evaluation_summary.json",
                )
                logger.info(f"[evaluation_main] Wrote paper-oriented summary to: {summary_out}")
            else:
                logger.info("[evaluation_main] No summary-compatible pillars requested; skipping evaluation summary build.")
        except Exception as e:
            logger.exception(f"[evaluation_main] Failed to build/write evaluation summary: {e}")

    logger.info(f"[evaluation_main_new] Done. Outputs at: {eval_root}")
    return eval_root