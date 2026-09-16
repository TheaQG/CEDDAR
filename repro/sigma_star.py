"""Prepare a small, isolated sigma* run, then explicitly generate or evaluate it."""
import argparse
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from sbgm.runtime import SOURCE_ROOT, external_output, setup_environment
from sbgm.sigma_control import build_edm_schedule, sigma_star_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "generate", "evaluate", "plot"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=SOURCE_ROOT / "sbgm/config/component_study/F_final_test_eval.yaml")
    parser.add_argument("--sigma-star-grid", nargs="+", type=float, default=[0.95, 1.0, 1.05])
    parser.add_argument("--sigma-star-mode", choices=["global", "late_ramp"], default="late_ramp")
    parser.add_argument("--initial-state", choices=["schedule", "legacy_sigma_max"], default="schedule")
    parser.add_argument("--noise-mode", choices=["sequential", "paired"], default="sequential")
    parser.add_argument("--seed", type=int, help="Sampler experiment seed; default is retained from config")
    parser.add_argument("--steps", type=int, default=56)
    parser.add_argument("--ensemble-size", type=int, default=2)
    parser.add_argument("--max-dates", type=int, default=1)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    args = parser.parse_args()
    if args.action != "prepare":
        prepare_flags = {'--config', '--sigma-star-grid', '--sigma-star-mode', '--initial-state',
                         '--steps', '--ensemble-size', '--max-dates', '--split', '--noise-mode', '--seed'}
        if any(token.split('=')[0] in prepare_flags for token in sys.argv[1:]):
            parser.error("Overrides apply only to prepare; generate/evaluate/plot use resolved_config.yaml")
    run = external_output(args.run_dir)
    saved = run / "resolved_config.yaml"
    if args.action == "prepare" and run.exists():
        parser.error("Prepare requires a new run directory; existing runs are never overwritten")
    if args.action != "prepare" and not saved.is_file():
        parser.error(f"Prepare the run first: missing {saved}")
    # Isolate outputs/caches even when smoke-test environment variables remain set.
    os.environ.update(CEDDAR_RUNS=str(run), SAMPLE_DIR=str(run / "samples"),
                      EVAL_DIR=str(run / "evaluation"), LOG_DIR=str(run / "logs"),
                      TMPDIR=str(run / "tmp"), XDG_CACHE_HOME=str(run / "cache"),
                      MPLCONFIGDIR=str(run / "cache/matplotlib"), TORCH_HOME=str(run / "cache/torch"))
    if args.action == "prepare":
        for key in ("DATA_DIR", "STATS_LOAD_DIR", "PUBLISHED_CHECKPOINT"):
            if not os.environ.get(key) or not Path(os.environ[key]).exists():
                parser.error(f"Set {key} to the existing input artifact")
        checkpoint = Path(os.environ['PUBLISHED_CHECKPOINT']).resolve()
        if not checkpoint.is_file():
            parser.error("PUBLISHED_CHECKPOINT must be a file")
        os.environ['CKPT_DIR'] = str(checkpoint.parent)
    setup_environment()
    from sbgm.utils import load_config
    if args.action == "prepare":
        cfg = load_config(str(args.config))
        # A supplied resolved config may contain absolute paths from another run.
        for key, value in dict(sample_dir=run / 'samples', evaluation_dir=run / 'evaluation',
                               log_dir=run / 'logs', path_save=run / 'samples').items():
            cfg.paths[key] = str(value)
        cfg.diagnostics.histogram_path = str(run / 'logs/histograms')
        cfg.full_gen_eval.gen_dir = None
        cfg.full_gen_eval.eval_dir = None
        cfg.paths.inference_checkpoint = str(checkpoint)
        cfg.training.device = os.environ.get('DEVICE', 'cpu')
        cfg.data_handling.num_workers = 0
        cfg.data_handling.pin_memory = False
        cfg.edm.sampling_steps = args.steps
        cfg.full_gen_eval.update(split=args.split, ensemble_size=args.ensemble_size,
                                 max_dates=args.max_dates, sigma_star_grid=args.sigma_star_grid)
        cfg.full_gen_eval.sigma_control.update(
            sigma_star_mode=args.sigma_star_mode, sigma_star_initial_state=args.initial_state,
            require_generation_manifest=True, example_sigma_subset=args.sigma_star_grid,
            noise_mode=args.noise_mode)
        if args.seed is not None:
            cfg.full_gen_eval.seed = args.seed
        if args.ensemble_size < 2 or args.max_dates == 0 or args.max_dates < -1:
            parser.error("Use ensemble-size >= 2 and max-dates >= 1 (or -1 for all)")
        if len({f'{v:.2f}' for v in args.sigma_star_grid}) != len(args.sigma_star_grid):
            parser.error("Grid values must be unique at two-decimal output precision")
        controls = sigma_star_kwargs(cfg.edm, cfg.full_gen_eval.sigma_control)
        for value in args.sigma_star_grid:
            _, schedule = build_edm_schedule(
                args.steps, **{k: cfg.edm[k] for k in ('sigma_min', 'sigma_max', 'rho', 'S_churn', 'S_min', 'S_max', 'S_noise')},
                **{**controls, 'sigma_star': value})
            print(f"sigma*={value}: mode={args.sigma_star_mode}, initial_std={schedule['initial_std']}, "
                  f"ramp=({schedule['ramp_start_index']},{schedule['ramp_end_index']})")
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), saved)
        print(f"Prepared {saved}; no checkpoint loaded and no samples generated.")
        return
    if args.action == 'plot':
        from sbgm.utils import get_model_string
        from sbgm.evaluate.evaluate_prcp.eval_sigma_star.evaluate_sigma_control import plot_saved_sigma_control
        cfg = load_config(str(saved))
        out_dir = Path(cfg.paths.evaluation_dir) / get_model_string(cfg) / 'prcp/sigma_control'
        plot_saved_sigma_control(cfg, out_dir)
        return
    # Later actions use only the saved, resolved configuration (not new overrides).
    from sbgm.cli.main_app import main as main_app
    sys.argv = [sys.argv[0], '--config_path', str(saved), '--mode',
                'sigma_star_generation' if args.action == 'generate' else 'sigma_star_evaluation', '--make_plots']
    main_app()


if __name__ == "__main__":
    main()
