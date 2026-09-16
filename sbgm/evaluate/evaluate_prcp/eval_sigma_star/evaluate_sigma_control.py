"""
Main entrypoint for σ*-dependent evaluation.
"""
import json
import numpy as np
import logging
from pathlib import Path
from sbgm.evaluate.evaluate_prcp.eval_sigma_star.metrics_sigma_control import evaluate_sigma_control
from sbgm.evaluate.evaluate_prcp.eval_sigma_star.plot_sigma_control import plot_sigma_control, plot_sigma_control_examples_grid, plot_sigma_control_psd_curves
from sbgm.utils import get_model_string
from sbgm.provenance import sigma_generation_metadata, write_provenance
from sbgm.sigma_control import sigma_star_kwargs
from sbgm.runtime import external_output

logger = logging.getLogger(__name__)

def run(cfg, make_plots=True):
    model_name = get_model_string(cfg)
    sigma_grid = cfg.full_gen_eval.sigma_star_grid
    base_gen = Path(cfg.paths.sample_dir) / "generation" / model_name
    out_dir = external_output(Path(cfg.paths.evaluation_dir) / model_name / "prcp" / "sigma_control")
    observed = sigma_generation_metadata(base_gen, sigma_grid, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read sigma_control config for possible subset of sigma* for examples/PSD
    scfg = getattr(getattr(cfg, "full_gen_eval", {}), "sigma_control", {}) if hasattr(cfg, "full_gen_eval") else {}

    logger.info(f"[SigmaControl] Evaluating σ* grid {sigma_grid} for {model_name}")

    metrics_paths = evaluate_sigma_control(cfg, sigma_grid, base_gen, out_dir)
    logger.info("[SigmaControl] Metrics written: %s", metrics_paths)

    # Label the band actually saved by the metric code, not an unverified request.
    psd_band = None
    psd_path = out_dir / 'tables/sigma_psd_curves.npz'
    if psd_path.is_file():
        with np.load(psd_path) as arrays:
            psd_band = arrays['psd_band_km'].tolist()

    # Figures may label a mode only when it was recorded at the sampler call.
    actual = observed[0]["sampler"] if observed and all(r["sampler"] for r in observed) else None
    meta = {
        "sigma_star_grid": [float(s) for s in sigma_grid],
        "requested_control": sigma_star_kwargs(cfg.edm, scfg),
        "generation": observed,
        "ramp": None if actual is None else {
            "mode": actual["sigma_star_mode"],
            "start_frac": actual["ramp_start_frac"], "end_frac": actual["ramp_end_frac"],
            "start_sigma": actual["ramp_start_sigma"], "end_sigma": actual["ramp_end_sigma"],
            "initial_state": actual["sigma_star_initial_state"],
        },
        "metrics": {"psd": "PSD of physical ensemble mean, averaged across dates",
                    "correlation": "low-pass PMM versus LR",
                    "crps_rain_thresh": scfg.get("crps_rain_thresh"),
                    "eval_land_only": bool(cfg.full_gen_eval.get("eval_land_only", True)),
                    "psd_band_km": psd_band,
                    "error_bars": "across-date standard deviation or nominal SEM; not ensemble spread"},
    }
    with (out_dir / "sigma_control_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    write_provenance(out_dir / "meta", cfg, stage="sigma_evaluation", generation=observed)

    if make_plots:
        plot_saved_sigma_control(cfg, out_dir)

    logger.info(f"[SigmaControl] Done. Results in {out_dir}")
    logger.info("[SigmaControl] Figures in: %s", str(Path(out_dir) / "figures"))
    return out_dir

def plot_saved_sigma_control(cfg, out_dir):
    """Regenerate figures from saved evaluation tables; no inference or metrics."""
    out_dir = Path(out_dir)
    summary = out_dir / 'tables/agg_summary.csv'
    if not summary.is_file():
        raise FileNotFoundError(summary)
    full = cfg.full_gen_eval
    subset = full.get('sigma_control', {}).get('example_sigma_subset')
    figures = out_dir / 'figures'
    plot_sigma_control(summary, figures, combined=bool(full.get('sigma_control_plot_combined', True)),
                       error_mode='sem', also_write_std=True)
    plot_sigma_control_psd_curves(out_dir, sigma_subset=subset)
    plot_sigma_control_examples_grid(
        cfg, sigma_star_grid=full.sigma_star_grid,
        gen_base_dir=Path(cfg.paths.sample_dir) / 'generation' / get_model_string(cfg),
        out_dir=out_dir, sigma_star_subset=subset,
        n_members=int(full.get('example_n_members', 3)), date=full.get('example_date'),
        land_only=bool(full.get('eval_land_only', True)), fname='examples_sigma_grid.png')
