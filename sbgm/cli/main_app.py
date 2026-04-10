""" 
    UNIFIED CLI INTERFACE FOR SBGM_SD

    main_app.py
    This script serves as the main control point for the full SBGM_SD application.
    Tasks implemented:
        - Running the training process  
        - Running the generation process on a trained model
        - Running the evaluation process from generated samples
        - Full model pipeline: training --> generation --> evaluation
        - 

    Tasks to be implemented:
        - Data structuring (train/test/eval splits)
        - Running full Dataset statistics based on config
"""
import argparse
import os
import logging
from datetime import timedelta
import faulthandler
faulthandler.enable()  # Enable faulthandler to get tracebacks on segfaults and timeouts

from omegaconf import OmegaConf

from sbgm.utils import get_model_string, load_config
from sbgm.logging_utils import (
    cfg_hash, make_run_name, ensure_run_dir,
    setup_logging, write_run_manifest, log_banner
)
from baselines.baseline_main import run as run_baselines
# from baselines.baseline_eval import run_all as run_baseline_eval
from baselines.evaluate_baselines.evaluation_baselines import run_all_baselines

import torch
import torch.distributed as dist


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _setup_runtime_context() -> dict:
    distributed_request = _env_flag("SBGM_DISTRIBUTED", False) or (os.environ.get("WORLD_SIZE") not in (None, "", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = distributed_request and world_size > 1

    # Allow long rank-0-only phases such as validation / plotting / generation
    # without tripping the default NCCL watchdog timeout.
    ddp_timeout_minutes = int(os.environ.get("SBGM_DDP_TIMEOUT_MINUTES", "120"))

    if is_distributed and not dist.is_initialized():
        backend = "nccl"
        if not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=ddp_timeout_minutes),
        )

    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device_name = f"cuda:{local_rank}"
        else:
            device_name = "cuda:0"
    else:
        device_name = "cpu"
    
    return {
        "distributed": is_distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main_process": (rank == 0),
        "device": device_name,
        "ddp_timeout_minutes": ddp_timeout_minutes,
    }


def _dist_barrier_if_needed(runtime: dict) -> None:
    if runtime.get("distributed", False) and dist.is_available() and dist.is_initialized():
        dist.barrier()

def _cfg_get(cfg, *keys, default=None):
    """Safely read nested config values from either DictConfig-like objects or plain dicts."""
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

def check_model_exists(cfg):
    model_name = get_model_string(cfg)
    checkpoint_dir = _cfg_get(cfg, "paths", "checkpoint_dir")
    if checkpoint_dir is None:
        raise RuntimeError("Config is missing paths.checkpoint_dir")
    ckpt_dir = os.path.join(checkpoint_dir, model_name + '.pth.tar')
    exists = os.path.exists(ckpt_dir)
    return exists, ckpt_dir

def check_generated_samples_exist(cfg):
    model_name = get_model_string(cfg)
    sample_dir = _cfg_get(cfg, "paths", "sample_dir")
    if sample_dir is None:
        raise RuntimeError("Config is missing paths.sample_dir")
    gen_dir = os.path.join(sample_dir, "generation", model_name, "generated_samples")
    exists = os.path.exists(gen_dir) and any(f.startswith("gen_samples") for f in os.listdir(gen_dir))
    return exists, gen_dir


# Use setup_logger and write_run_manifest in main
def main():
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="SBGM full pipeline launcher")
    parser.add_argument("--config_path", required=True, help="Path to the yaml config")
    
    parser.add_argument(
        "--mode",
        choices=[
            "train", "generate", "evaluate", "eval2", "full_pipeline",
            "data_splits", "quicklook", "baseline", "baseline_eval",
            "sigma_star_generation", "sigma_star_evaluation",
            "sampler_grid_generation", "sampler_grid_evaluation",  # <-- add these
        ],
        default="full_pipeline"
    )
    
    parser.add_argument("--baseline_type", choices=["bilinear", "qm", "unet_sr"], default="bilinear", help="If mode is 'baseline', which baseline to run.")
    parser.add_argument("--baseline_split", choices=["train", "valid", "test"], default="test", help="If mode is 'baseline', which split to use.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--make_plots", action="store_true", help="If set, make publication-ready plots after evaluation.")
    parser.add_argument("--dry_run", action="store_true", help="If set, no actual training/generation/evaluation will be performed, only config parsing and logging setup.")
    args = parser.parse_args()


    from omegaconf import DictConfig, ListConfig
    from typing import cast

    cfg = load_config(args.config_path)
    # Ensure cfg is a DictConfig (some loaders may return a ListConfig); accept a single-element ListConfig wrapping a DictConfig
    if isinstance(cfg, ListConfig):
        if len(cfg) == 1 and isinstance(cfg[0], DictConfig):
            cfg = cfg[0]
        else:
            raise RuntimeError("Expected a DictConfig or a single-element ListConfig containing a DictConfig for cfg.")
    cfg = cast(DictConfig, cfg)
    runtime = _setup_runtime_context()

    # === Build run context ===
    model_name = get_model_string(cfg)
    h = cfg_hash(cfg)
    run_name = make_run_name(cfg.experiment.name, h)
    run_dir = ensure_run_dir(cfg.paths.log_dir, model_name)

    make_plots = args.make_plots or (args.mode in ['evaluate', 'eval2', 'full_pipeline'] and not args.skip_evaluation)

    # === Logging + manifest ===
    cfg_py = OmegaConf.to_container(cfg, resolve=True) # Convert to plain dict for logging
    file_level = getattr(cfg, "logging", {}).get("file_level", "INFO")
    console_level = getattr(cfg, "logging", {}).get("console_level", "WARNING")
    log_path = setup_logging(
        run_dir,
        run_name,
        file_level=getattr(cfg, "logging", {}).get("file_level", "INFO"),
        console_level=getattr(cfg, "logging", {}).get("console_level", "WARNING")
    )

    if runtime["is_main_process"]:
        logger.info("Unified log file: %s", log_path)
        write_run_manifest(run_dir, run_name, cfg, model_name)
    
    logger.info("=== ENTERED SBGM_SD MAIN APP ===")
    logger.info("Experiment      : %s", cfg.experiment.name)
    logger.info("Mode            : %s", args.mode)
    logger.info("Config          : %s", args.config_path)
    logger.info("Run dir         : %s", run_dir)
    logger.info("Log file        : %s", log_path)
    logger.info("Model key       : %s", model_name)
    logger.info("Cfg hash        : %s", h)
    logger.info(
        "Runtime         : distributed=%s rank=%s local_rank=%s world_size=%s device=%s ddp_timeout_minutes=%s",
        runtime["distributed"],
        runtime["rank"],
        runtime["local_rank"],
        runtime["world_size"],
        runtime["device"],
        runtime.get("ddp_timeout_minutes"),
    )

    # Imports kept here to avoid circular imports
    from sbgm.cli import (
        launch_sbgm,
        launch_generation,
        launch_evaluation,
        launch_quicklook,
        launch_generation_sigma_star,
        launch_evaluation_sigma_star,
        launch_generation_sampler_grid,
        launch_evaluation_sampler_grid,
    )
    from data_analysis_pipeline.cli import launch_split_creation

    # === Dispatch with banners ===
    cfg_run = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if not isinstance(cfg_run, dict):
        raise RuntimeError("Resolved configuration must be a dictionary-like object.")
    cfg_run.setdefault("runtime", {})
    cfg_run["runtime"].update(runtime)

    if args.mode == "data_splits":
        log_banner("DATA SPLIT CREATION START")
        launch_split_creation.run(cfg_run)
        log_banner("DATA SPLIT CREATION DONE")

    if args.mode == "train":
        log_banner("TRAINING START")
        launch_sbgm.run_training(cfg_run)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("TRAINING DONE")

    elif args.mode == "generate":
        if runtime["is_main_process"]:
            log_banner("GENERATION START")
        launch_generation.run_generation(cfg_run)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("GENERATION DONE")

    elif args.mode == "evaluate":
        if runtime["is_main_process"]:
            log_banner("EVALUATION START")
        launch_evaluation.run_evaluation(cfg_run, make_plots=make_plots)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("EVALUATION DONE")

    elif args.mode == "eval2":
        if runtime["is_main_process"]:
            log_banner("EVALUATION (EVAL2) START")
        launch_evaluation.run_evaluation(cfg_run, make_plots=make_plots, force_eval2=True)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("EVALUATION (EVAL2) DONE")

    elif args.mode == "quicklook":
        if runtime["is_main_process"]:
            log_banner("QUICKLOOK START")
            exists, ckpt_dir = check_model_exists(cfg_run)
            if not exists:
                raise RuntimeError(f"Cannot run quicklook: model checkpoint not found in {ckpt_dir}")
            launch_quicklook.run_quicklook(cfg_run)
            log_banner("QUICKLOOK DONE")
        _dist_barrier_if_needed(runtime)

    elif args.mode == "full_pipeline":
        if runtime["is_main_process"]:
            log_banner("TRAINING START")
        exists, ckpt_dir = check_model_exists(cfg_run)
        if args.skip_train and not exists:
            raise RuntimeError(f"Cannot skip training: no trained model found in {ckpt_dir}")
        if not args.skip_train:
            launch_sbgm.run_training(cfg_run)
        _dist_barrier_if_needed(runtime)
    
        # DDP must be torn down before rank-0-only serial stages, otherwise non-main ranks will sit
        # in collectives/barriers while rank 0 performs lon-running quicklook/generation/evaluation.
        if runtime.get("distributed", False) and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        if not runtime["is_main_process"]:
            logger.info("Non-main rank exiting after distributed training; serial pipeline continues on rank 0 only.")
            return
        
        log_banner("TRAINING DONE")

        log_banner("QUICKLOOK START")
        exists, ckpt_dir = check_model_exists(cfg_run)
        if not exists:
            raise RuntimeError(f"Cannot run quicklook: model checkpoint not found in {ckpt_dir}")
        launch_quicklook.run_quicklook(cfg_run)
        log_banner("QUICKLOOK DONE")

        log_banner("GENERATION START")
        exists, gen_dir = check_generated_samples_exist(cfg_run)
        if not args.skip_generation:
            launch_generation.run_generation(cfg_run)
        log_banner("GENERATION DONE")

        log_banner("EVALUATION START")
        if not args.skip_evaluation:
            launch_evaluation.run_evaluation(cfg_run, make_plots=make_plots)
        log_banner("EVALUATION DONE")

    elif args.mode == "baseline":
        if runtime["is_main_process"]:
            log_banner("BASELINE START")
            run_baselines(cfg_run)
            log_banner("BASELINE DONE")
        _dist_barrier_if_needed(runtime)

    elif args.mode == "baseline_eval":
        if runtime["is_main_process"]:
            log_banner("BASELINE EVALUATION START")
            # run_baseline_eval(cfg)
            run_all_baselines(cfg_run)
            log_banner("BASELINE EVALUATION DONE")
        _dist_barrier_if_needed(runtime)

    elif args.mode == "sigma_star_generation":
        if runtime["is_main_process"]:
            log_banner("SIGMA_STAR GENERATION START")
        launch_generation_sigma_star.run(cfg_run)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("SIGMA_STAR GENERATION DONE")

    elif args.mode == "sigma_star_evaluation":
        if runtime["is_main_process"]:
            log_banner("SIGMA_STAR EVALUATION START")
        # use args.make_plots to also toggle making qualitative example montages
        launch_evaluation_sigma_star.run(cfg_run, make_plots=make_plots, make_examples=args.make_plots)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("SIGMA_STAR EVALUATION DONE")

    elif args.mode == "sampler_grid_generation":
        if runtime["is_main_process"]:
            log_banner("SAMPLER GRID GENERATION START")
        launch_generation_sampler_grid.run(cfg_run)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("SAMPLER GRID GENERATION DONE")

    elif args.mode == "sampler_grid_evaluation":
        if runtime["is_main_process"]:
            log_banner("SAMPLER GRID EVALUATION START")
        # make_plots can control whether the evaluation makes plots or only tables
        launch_evaluation_sampler_grid.run(cfg_run, make_plots=make_plots)
        _dist_barrier_if_needed(runtime)
        if runtime["is_main_process"]:
            log_banner("SAMPLER GRID EVALUATION DONE")

    logger.info("=== SBGM_SD MAIN APP DONE ===")
    if runtime.get("distributed", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()