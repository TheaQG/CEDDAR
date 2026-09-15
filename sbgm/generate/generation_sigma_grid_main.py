from sbgm.provenance import checkpoint_info
import os
import logging
from pathlib import Path
import numpy as np
import torch

from sbgm.training_utils import get_model, get_final_gen_dataloader
from sbgm.generate.generation import GenerationRunner, GenerationConfig
from sbgm.utils import get_model_string
from sbgm.sigma_control import build_edm_schedule, sigma_star_kwargs
from sbgm.runtime import external_output

logger = logging.getLogger(__name__)

def _resolve_base_out_dir(cfg) -> Path:
    """Base dir: <paths.sample_dir>/generation/<model_name>/ (robust to dict/attr cfg)."""
    model_name_str = get_model_string(cfg)
    # Try attribute access first, fall back to dict-style
    sample_dir = None
    if hasattr(cfg, "paths") and hasattr(cfg.paths, "sample_dir"):
        sample_dir = cfg.paths.sample_dir
    elif isinstance(cfg, dict) and "paths" in cfg and "sample_dir" in cfg["paths"]:
        sample_dir = cfg["paths"]["sample_dir"]
    else:
        raise KeyError("Could not resolve cfg.paths.sample_dir")
    base = external_output(Path(sample_dir) / "generation" / model_name_str)
    base.mkdir(parents=True, exist_ok=True)
    return base

def _build_generation_config(cfg, out_root: Path) -> GenerationConfig:
    cfg_full_gen_eval = cfg.get('full_gen_eval', cfg)
    M = int(cfg_full_gen_eval.get('ensemble_size', cfg.data_handling.get('n_gen_samples', 32)))
    edm = cfg.get('edm', {})

    return GenerationConfig(
        output_root=str(out_root),
        ensemble_size=M,
        sampler_steps=int(edm.get('sampling_steps', 40)),
        seed=int(cfg_full_gen_eval.get('seed', 1234)),
        use_edm=bool(edm.get('enabled', True)),
        sigma_min=float(edm.get('sigma_min', 0.002)),
        sigma_max=float(edm.get('sigma_max', 80.0)),
        rho=float(edm.get('rho', 7.0)),
        S_churn=float(edm.get('S_churn', 0.0)),
        S_min=float(edm.get('S_min', 0.0)),
        S_max=float(edm.get('S_max', float('inf'))),
        S_noise=float(edm.get('S_noise', 1.0)),
        predict_residual=bool(edm.get('predict_residual', False)),
        save_space="physical",
        max_dates=int(cfg_full_gen_eval.get('max_dates', -1)),
    )


def generation_sigma_grid_main(cfg):
    """
    Generate ensembles across a grid of sigma_star values.
    For each sigma_star, outputs go to:
      <sample_dir>/generation/<model_name>/sigma_star=<val>/
    """
    full = cfg.get('full_gen_eval', {})
    edm = cfg.get('edm', {})
    controls = sigma_star_kwargs(edm, full.get('sigma_control', {}))
    values = full.get('sigma_star_grid', [1.0])
    grid = [float(values)] if isinstance(values, (int, float)) else [float(v) for v in values]
    names = [f"sigma_star={v:.2f}" for v in grid]
    if not grid or len(set(names)) != len(grid):
        raise ValueError("sigma_star_grid must be nonempty and unique at two-decimal output precision")
    # Validate every trajectory before loading weights or generating any grid point.
    for value in grid:
        build_edm_schedule(
            num_steps=int(edm.get('sampling_steps', 40)),
            **{k: edm[k] for k in ('sigma_min', 'sigma_max', 'rho', 'S_churn', 'S_min', 'S_max', 'S_noise') if k in edm},
            **{**controls, 'sigma_star': value},
        )
    base_out = _resolve_base_out_dir(cfg)
    if any((base_out / name).exists() for name in names):
        raise FileExistsError(f"Existing sigma* outputs under {base_out}; use a fresh SAMPLE_DIR")

    # ----------------------- Seed -----------------------
    seed = int(getattr(getattr(cfg, "full_gen_eval", {}), "seed", 1234) if not isinstance(cfg, dict) else cfg.get("full_gen_eval", {}).get("seed", 1234))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # ----------------------- Device -----------------------
    device = getattr(getattr(cfg, "training", {}), "device", None)
    if device is None and isinstance(cfg, dict):
        device = cfg.get("training", {}).get("device", "cpu")
    if device is None:
        device = "cpu"

    # ----------------------- Model & checkpoint -----------------------
    model, ckpt_dir, ckpt_name = get_model(cfg)
    ckpt_path = cfg.get("paths", {}).get("inference_checkpoint") or os.path.join(ckpt_dir, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "network_params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'network_params': {ckpt_path}")
    model.load_state_dict(ckpt["network_params"])
    model._ceddar_checkpoint = checkpoint_info(ckpt_path, "network_params")
    model.eval()
    logger.info(f"[generation_sigma_grid_main] Loaded checkpoint: {ckpt_path}")

    # ----------------------- Data (deterministic, split-aware) -----------------------
    # Decide which temporal split to generate for: train / val / test
    if isinstance(cfg, dict):
        full_gen_eval = cfg.get("full_gen_eval", {})
        split_cfg = str(full_gen_eval.get("split", "test")).lower()
    else:
        full_gen_eval = getattr(cfg, "full_gen_eval", {})
        split_cfg = str(getattr(full_gen_eval, "split", "test")).lower()

    if split_cfg in ("val", "valid", "validation"):
        split_for_dataset = "valid"
    elif split_cfg == "train":
        split_for_dataset = "train"
    else:
        split_for_dataset = "test"

    # Make sure data_handling exists
    if isinstance(cfg, dict):
        cfg.setdefault("data_handling", {})
        dh = cfg["data_handling"]
        dh["split"] = split_for_dataset
        dh["shuffle"] = False
        dh["drop_last"] = False
    else:
        if not hasattr(cfg, "data_handling") or cfg.data_handling is None:
            cfg.data_handling = {}
        cfg.data_handling["split"] = split_for_dataset
        cfg.data_handling["shuffle"] = False
        cfg.data_handling["drop_last"] = False

    logger.info(f"[generation_sigma_grid_main] Using data split='{split_for_dataset}' for sigma* grid generation")

    gen_dataloader = get_final_gen_dataloader(cfg, split=split_for_dataset)

    # Keep the original sequential RNG stream; the manifest records this scope.
    for sstar in grid:
        cfg['edm'].update(controls)
        cfg['edm']['sigma_star'] = sstar
        subdir = base_out / f"sigma_star={sstar:.2f}"
        # No resume/overwrite: a directory must represent one sampler invocation.
        subdir.mkdir(exist_ok=False)
        logger.info("[generation_sigma_grid_main] sigma*=%s, controls=%s", sstar, controls)
        gen_cfg = _build_generation_config(cfg, subdir)
        runner = GenerationRunner(model=model, cfg=cfg, device=device, out_root=subdir, gen_config=gen_cfg)
        runner.run(gen_dataloader)

    logger.info(f"[generation_sigma_grid_main] Done. Outputs at: {base_out}")
    return base_out