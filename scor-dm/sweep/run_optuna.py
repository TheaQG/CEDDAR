#! /usr/bin/env python3
"""
    run_optuna.py - Bayesian hyperparameter optimization for the SBGM for climate variable downscaling
    -------------------------------------------------------------
    - Works with TrainingPipeline_general in `sbgm_training.py`
    - Robust against architecture mismatches (prunes trials that break model build)
    - Adds project-root to `sys.path`so import like ìmport sbgm.` succeed when the script lives in `sbgm/sweep/`
    - Search space tuned to acoid invalid attention head combos
    - Generates a concrete YAML for every trial under `sbgm/sweep/generated/`

    Usage (local smoke test CPU):
        python run_optuna.py --n-trials 3 --study-name sbgm_baseline \
            --enable-medium True --epochs 1

    Usage (SLURM 200-trial array on LUMI (1 trial per task)):
    sbatch --array=0-199 run_optuna.sh \
        --n-trials 200 --study-name sbgm_lumi_200 \
        --enable-medium True --epochs 1
    (Requires `run_optuna.sh` to set the environment variables)
    -------------------------------------------------------------
"""

from __future__ import annotations  # For type hinting in Python 3.7+

import argparse
import yaml
import pathlib

import torch
import re
import os
import sys
import warnings
import tempfile

import random
import numpy as np

from datetime import datetime

from functools import partial

import optuna
from optuna.samplers import GPSampler
from optuna.pruners import SuccessiveHalvingPruner

import logging
logger = logging.getLogger(__name__)



# ── Make the project root importable ────────────────────────────────
THIS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT  = THIS_DIR.parent.parent                            # …/SBGM_SD
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence optuna list->tuple warnings
warnings.filterwarnings(
    "ignore",
    category= UserWarning,
    message=r"Choices for a categorical distribution. *type list*",
)

# ==========================================================
# Paths and constants, and helpers
# ==========================================================
def _expand_env(obj):
    """ Recursively replace ${env:VAR} with the value of $VAR."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        # ${env:DATA_DIR}  →  /absolute/path
        return re.sub(r"\${env:([^}]+)}",
                      lambda m: os.environ.get(m.group(1), m.group(0)),
                      obj)
    return obj
    

# Base dir is this file's parent directory
BASE_DIR = THIS_DIR
# Generated configs will live in PROJECT_ROOT/sbgm/sweep/generated/
GENERATED_CFG_DIR = BASE_DIR / "generated"
GENERATED_CFG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


# ==========================================================
# Utilities
# ==========================================================

# Fixes reproducibility inside each trial
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Flattened-key update utility (e.g. "training.learning_rate")
def deep_update(cfg: dict, flat_kv: dict):
    for flat_key, value in flat_kv.items():
        node = cfg  # Start at the root of the config
        *parents, leaf = flat_key.split('.')
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value  # Set the final value in the nested structure

# ===========================================================
# Search-space helpers - to be tweaked
# ===========================================================

def sample_high_impact(trial: optuna.trial.Trial) -> dict:
    """
        Sample high-impact hyperparameters for the SBGM.
        Return a dict {flattened.key: value} with the highest-impact hyperparameters.
    """
    return {
        # Optimiser/learning dynamics
        "training.leargning_rate":      trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "training.optimizer":           trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        # Diffusion schedule and sampling
        "sampler.n_timesteps":          trial.suggest_int("n_steps", 200, 1500),
        "sampler.time_embedding":       trial.suggest_int("t_embed", 128, 512, step=64),
        # Classifier-free guidance
        "classifier_free_guidance.guidance_scale": trial.suggest_float("cfg_scale", 0.5, 6.0, log=True),
        # Score-UNet architecture
        "sampler.block_layers":         trial.suggest_categorical("block_layers", [(2, 2, 2, 2), (3, 3, 3, 3), (3, 4, 6, 3)]),
        # Only divisors of common fmap sizes (32, 64, 128,...)
        "sampler.num_heads":            trial.suggest_categorical("n_heads", [1, 2, 4, 8, 16]),
    }

def sample_medium_impact(trial: optuna.trial.Trial) -> dict:
    """
        Sample medium-impact hyperparameters for the SBGM.
        Return a dict {flattened.key: value} with the medium-impact hyperparameters.
    """
    return {
        "training.batch_size":          trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "training.ema_decay":           trial.suggest_float("ema_decay", 0.95, 0.9999),
        "training.weight_decay":        trial.suggest_float("wd", 0.0, 1e-3, log=True),
        "sampler.last_fmap_channels":   trial.suggest_categorical("last_fmap_channels", [256, 512, 768]),
    }



# ===========================================================
# Utility - merge dot-notation keys into nested dict
# ===========================================================

def compose_cfg(base_cfg: dict, flat_params:dict) -> dict:
    """
        Insert flattened.dot.keys into a nested YAML structure (in-place copy)
    """

    import copy
    
    cfg = copy.deepcopy(base_cfg)  # Start with a copy of the base config
    for key, value in flat_params.items():
        node = cfg # Start at the root of the config
        *path, last = key.split('.') # Split the key into path components
        for p in path:
            node = node.setdefault(p, {})
        node[last] = value  # Set the final value in the nested structure
    return cfg

# ===========================================================
# Optuna objective - one *trial*
# ===========================================================

def objective(trial: optuna.trial.Trial,
              enable_medium: bool = False,
              n_epochs: int | None = None) -> float:
    
    # 1. Assemble a concrete config dict
    cfg = _expand_env(
        yaml.safe_load(open(PROJECT_ROOT / "sbgm/config/default_config.yaml"))
    )

    # cfg = yaml.safe_load(open(PROJECT_ROOT / "sbgm/config/default_config.yaml"))
    deep_update(cfg, sample_high_impact(trial))  # Always sample high-impact
    if enable_medium:
        deep_update(cfg, sample_medium_impact(trial))

    # Allow per-trial shortening of epochs for multi-fidelity pruning
    if n_epochs is not None:
        cfg['training']['epochs'] = n_epochs

    # 2. Persist concrete YAML for reproducibility
    out_cfg = GENERATED_CFG_DIR / f"trial_{trial.number:05d}.yaml"
    yaml.safe_dump(cfg, open(out_cfg, 'w'))

    # 3. Reproducibility: set seeds
    set_seed(cfg["training"].get("seed", 42) + trial.number)

    # 4. Build a model, data, optimiser, and training pipeline - prune gracefully on assertion errors

    try:
        from sbgm.score_unet import ScoreNet, loss_fn, marginal_prob_std_fn, diffusion_coeff_fn
        from sbgm.training_utils import get_dataloader, get_optimizer, get_scheduler, get_model
        from sbgm.training import TrainingPipeline_general

        # Get the model with the helper
        model, checkpoint_path, checkpoint_name = get_model(cfg)
        device = cfg['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Set optimizer and scheduler
        optimizer = get_optimizer(cfg, model)
        
        # Get the learning rate scheduler (if applicable)
        lr_scheduler_type = cfg['training'].get('lr_scheduler', None)
        
        if lr_scheduler_type is not None:
            logger.info(f"▸ Using learning rate scheduler: {lr_scheduler_type}")
            scheduler = get_scheduler(cfg, optimizer)
        else:
            scheduler = None
            logger.info("▸ No learning rate scheduler specified, using default learning rate.")


        # Load data
        train_dataloader, val_dataloader, gen_dataloader = get_dataloader(cfg)

        # Define the training pipeline
        pipeline = TrainingPipeline_general(model=model,
                                            loss_fn=loss_fn,
                                            marginal_prob_std_fn=marginal_prob_std_fn,
                                            diffusion_coeff_fn=diffusion_coeff_fn,
                                            optimizer=optimizer,
                                            device=device,
                                            lr_scheduler=scheduler,
                                            cfg=cfg
                                            )
    except AssertionError as e:
        trial.set_user_attr("build_error", str(e))
        logger.error(f"▸ Trial {trial.number} failed to build model: {e}")
        raise optuna.TrialPruned()

    # 5. Train
    _, val_loss = pipeline.train(train_dataloader,
                                val_dataloader,
                                gen_dataloader,
                                cfg,
                                epochs=cfg['training']['epochs'],
                                verbose=cfg["training"].get('verbose', True),
                                use_mixed_precision= cfg['training'].get('use_mixed_precision', False)
    )

    # 6. Report to Optuna (lower is better). Enable pruning (i.e. early stopping) after each epoch
    trial.report(val_loss, step=cfg['training']['epochs'])
    logger.info(f"▸ Trial {trial.number} completed with validation loss: {val_loss:.4f}")
    if trial.should_prune():
        logger.warning(f"▸ Trial {trial.number} pruned at epoch {cfg['training']['epochs']} with loss {val_loss:.4f}")
        raise optuna.TrialPruned(f"Trial {trial.number} pruned at epoch {cfg['training']['epochs']} with loss {val_loss:.4f}")
    
    return val_loss

    

# ===========================================================
# Main function to run the optimization
# ===========================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--study-name", default="sbgm_optuna_test")
    p.add_argument("--storage", default="sqlite:///sbgm_optuna.db")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--enable-medium", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="Override cfg.training.epochs")

    args = p.parse_args()

    sampler = optuna.samplers.GPSampler(seed=42)  # Use a fixed seed for reproducibility
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3)

    study = optuna.create_study(direction="minimize",
                                study_name=args.study_name,
                                storage=args.storage,
                                load_if_exists=True,
                                sampler=sampler,
                                pruner=pruner)
    
    # Partial so that Optuna only sees trial

    study.optimize(partial(objective,
                           enable_medium=args.enable_medium,
                           n_epochs=args.epochs),
                    n_trials=args.n_trials)