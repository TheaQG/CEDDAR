"""
Extract EDM sampling trajectories for Noise-to-rain sonification/video notebook.

This script is intended to run on LUMI, where the trained checkpoint and full test data
are available. It performs inference only and writes compact .npz trajectory packages
that can be copied locally and post-processed in NoiseToRain.ipynb.

Design principle:
  - follow the normal generation/training path as closely as possible;
  - build the model with training_utils.get_model(cfg);
  - build data with training_utils.get_dataloader(cfg);
  - mirror GenerationRunner/TrainingPipeline_general conditioning logic;
  - only add sampler trajectory capture as an extra output.

Example:
    python -m sbgm.tools.sonify_sampling \
      --config /scratch/project_465002493/quistgaa/Code/CEDDAR/sbgm/config/paper2/P0.yaml \
      --output-dir /scratch/project_465002493/quistgaa/Code/CEDDAR/models_and_samples/noise_to_rain/V0 \
      --split test \
      --num-samples 4 \
      --members-per-sample 1 \
      --start-index 0 \
      --seed 504 \
      --capture-every 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from sbgm.score_sampling import edm_sampler
from sbgm.training_utils import get_dataloader, get_model
from sbgm.utils import get_model_string
try:
    from sbgm.generation import _build_back_transforms # type: ignore
except ImportError:
    from sbgm.generate.generation import _build_back_transforms



logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )


def _resolve_env_placeholders(obj: Any) -> Any:
    """Resolve simple ${env:VAR} strings when running outside the normal launcher."""
    if isinstance(obj, dict):
        return {k: _resolve_env_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_placeholders(v) for v in obj]
    if isinstance(obj, str):
        pattern = re.compile(r"\$\{env:([^}]+)\}")

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            value = os.environ.get(name)
            if value is None:
                logger.warning("Environment variable %s is not set; leaving placeholder unchanged.", name)
                return match.group(0)
            return value

        return pattern.sub(repl, obj)
    return obj

def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_placeholders(cfg)

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_get(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _model_ref(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _normalise_state_dict(raw_state):
    state = {}
    for key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        key = str(key)
        if key.startswith("module."):
            key = key[len("module."):]
        state[key] = value
    return state


def _select_checkpoint_state(raw, cfg):
    training_cfg = cfg.get("training", {}) or {}
    use_ema = bool(training_cfg.get("with_ema", False)) and (
        bool(training_cfg.get("eval_use_ema", False)) or bool(training_cfg.get("load_ema", False))
    )

    if use_ema and isinstance(raw.get("ema_network_params"), dict):
        return _normalise_state_dict(raw["ema_network_params"]), "ema_network_params"

    if isinstance(raw.get("network_params"), dict):
        return _normalise_state_dict(raw["network_params"]), "network_params"

    for key in ("model_state_dict", "state_dict", "model", "net", "score_model", "network"):
        if isinstance(raw.get(key), dict):
            return _normalise_state_dict(raw[key]), key

    direct = _normalise_state_dict(raw)
    if direct:
        return direct, "raw_state_dict"

    raise KeyError(f"Could not find model weights. Checkpoint keys: {list(raw.keys())}")



def _checkpoint_path_from_cfg(cfg: dict[str, Any]) -> Path:
    """
    Resolve checkpoint path from the configuration, matching the training pipeline.

    The training code saves checkpoints as:
        paths.checkpoint_dir / (get_model_string(cfg) + ".pth.tar")

    We intentionally do not accept a separate --checkpoint argument, because pairing an
    arbitrary config with an arbitrary checkpoint is an easy way to create silent model/data
    mismatches.
    """
    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_name = f"{get_model_string(cfg)}.pth.tar"
    return checkpoint_dir / checkpoint_name


def _load_model_from_cfg_checkpoint(cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, Path]:
    """Build model through training_utils.get_model(cfg), then load its config-derived checkpoint."""
    model, _, _ = get_model(cfg)
    model = model.to(device)

    checkpoint_path = _checkpoint_path_from_cfg(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Config-derived checkpoint does not exist: {checkpoint_path}\n"
            f"This path is derived from paths.checkpoint_dir and get_model_string(cfg). "
            f"Check that the YAML matches the trained model you want to sonify."
        )

    raw = torch.load(checkpoint_path, map_location="cpu")
    state, state_source = _select_checkpoint_state(raw, cfg)

    logger.info("Loading checkpoint weights from key '%s' with %d tensors", state_source, len(state))

    model.load_state_dict(state, strict=True)

    model.eval()
    return model, checkpoint_path


def _build_cond_img(batch: dict[str, Any], cfg: dict[str, Any], model: torch.nn.Module, device: torch.device) -> torch.Tensor | None:
    """
    Mirror GenerationRunner._build_cond_img / TrainingPipeline_general._build_cond_img.

    Returns the exact conditioning tensor passed to the U-Net:
      - local LR only if context encoder disabled;
      - context only if paper2.spatial_context.encoder.input_mode == 'context_only';
      - local LR + encoded large-domain context otherwise.
    """
    lr_vars = list(cfg.get("lowres", {}).get("condition_variables", []) or [])
    if len(lr_vars) == 0:
        return None

    paper2 = cfg.get("paper2", {}) or {}
    spatial = paper2.get("spatial_context", {}) or {}
    mode = str(spatial.get("mode", "")).lower()

    lr_tensors_local: list[torch.Tensor] = []
    for var in lr_vars:
        key_local = f"{var}_lr_local"
        key_ctx = f"{var}_lr"

        if mode == "large_domain":
            if key_local not in batch or batch[key_local] is None:
                raise KeyError(
                    f"paper2.spatial_context.mode='large_domain' requires '{key_local}' in batch. "
                    f"Available keys: {list(batch.keys())}"
                )
            t = batch[key_local]
        else:
            if key_local in batch and batch[key_local] is not None:
                t = batch[key_local]
            else:
                if key_ctx not in batch:
                    raise KeyError(f"Could not find '{key_ctx}' in batch. Available keys: {list(batch.keys())}")
                t = batch[key_ctx]

        t = _to_device_tensor(t, device)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        lr_tensors_local.append(t)

    cond_local = torch.cat(lr_tensors_local, dim=1).to(device)

    enc_cfg = spatial.get("encoder", {}) or {}
    model_ref = _model_ref(model)
    use_ctx = bool(enc_cfg.get("enabled", False)) and getattr(model_ref, "context_encoder", None) is not None
    ctx_mode = str(enc_cfg.get("input_mode", "context_plus_local"))
    if not use_ctx:
        return cond_local

    xs: list[torch.Tensor] = []
    for var in lr_vars:
        key = f"{var}_lr"
        if key not in batch:
            raise KeyError(
                f"Context encoder is enabled but full/context LR key '{key}' is missing. "
                f"Available keys: {list(batch.keys())}"
            )
        t = _to_device_tensor(batch[key], device)
        if t.ndim == 4:
            xs.append(t[:, 0])
        elif t.ndim == 3:
            xs.append(t)
        else:
            raise ValueError(f"Expected '{key}' to have shape [B,1,H,W] or [B,H,W], got {tuple(t.shape)}")

    x_bvhw = torch.stack(xs, dim=1)
    x_ctx = x_bvhw.unsqueeze(1).to(device)
    ctx = model_ref.encode_spatial_context(x_ctx)

    if ctx_mode == "context_only":
        return ctx
    return torch.cat([cond_local, ctx], dim=1)


def _build_local_cond_img(batch: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    """
    Mirror GenerationRunner/TrainingPipeline local LR conditioning.

    This excludes encoded context and is saved for human-readable plotting in the notebook.
    """
    lr_vars = list(cfg.get("lowres", {}).get("condition_variables", []) or [])
    if len(lr_vars) == 0:
        return None

    paper2 = cfg.get("paper2", {}) or {}
    spatial = paper2.get("spatial_context", {}) or {}
    mode = str(spatial.get("mode", "")).lower()

    lr_tensors_local: list[torch.Tensor] = []
    for var in lr_vars:
        key_local = f"{var}_lr_local"
        key_ctx = f"{var}_lr"

        if mode == "large_domain":
            key = key_local
        elif key_local in batch and batch[key_local] is not None:
            key = key_local
        else:
            key = key_ctx

        if key not in batch:
            raise KeyError(f"Missing local LR condition key '{key}'. Available keys: {list(batch.keys())}")

        t = _to_device_tensor(batch[key], device)
        if t.ndim == 3:
            t = t.unsqueeze(1)  # Add channel dim if missing
        elif t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)  # Add channel and spatial dims if missing
        lr_tensors_local.append(t)

    return torch.cat(lr_tensors_local, dim=1).to(device)


def _optional_tensor(batch: dict[str, Any], candidates: tuple[str, ...], device: torch.device) -> torch.Tensor | None:
    for key in candidates:
        if key in batch and batch[key] is not None:
            return _to_device_tensor(batch[key], device)
    return None


def _build_lr_baseline(batch: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    """Only pass lr_ups to edm_sampler when residual prediction is actually enabled."""
    if not bool(cfg.get("edm", {}).get("predict_residual", False)):
        return None
    return _optional_tensor(
        batch,
        (
            "lr_ups",
            "lr_baseline",
            "baseline",
            "prcp_lr_ups",
            "prcp_hrspace_lr",
            "prcp_lr_hrspace",
        ),
        device,
    )


def _extract_batch_item(batch: dict[str, Any], local_idx: int) -> dict[str, Any]:
    single: dict[str, Any] = {}

    for key, value in batch.items():

        # These are crop-coordinate bookkeeping fields from the dataset. PyTorch
        # collates them as a list of coordinate components, not as a list over
        # batch samples, so they are awkward to slice sample-by-sample. They are
        # not used by the sonification sampler or saving logic, so skip them.
        if key in ("hr_points", "lr_points"):
            continue

        if torch.is_tensor(value):
            single[key] = value[local_idx : local_idx + 1]

        elif isinstance(value, (list, tuple)):

            if local_idx >= len(value):

                logger.error(
                    "Batch extraction failed for key='%s': "
                    "local_idx=%s len(value)=%s type=%s",
                    key,
                    local_idx,
                    len(value),
                    type(value),
                )

                logger.error("Full batch structure:")

                for k, v in batch.items():

                    if torch.is_tensor(v):
                        logger.error(
                            "  %s -> tensor shape=%s",
                            k,
                            tuple(v.shape),
                        )

                    elif isinstance(v, (list, tuple)):
                        logger.error(
                            "  %s -> list len=%s",
                            k,
                            len(v),
                        )

                    else:
                        logger.error(
                            "  %s -> type=%s",
                            k,
                            type(v),
                        )

                raise IndexError(
                    f"local_idx={local_idx} exceeds len({key})={len(value)}"
                )

            single[key] = [value[local_idx]]

        else:
            single[key] = value

    return single


def _infer_batch_size(batch: dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
    raise RuntimeError("Could not infer batch size from dataloader batch.")


def _find_sample_id(batch: dict[str, Any], fallback: str) -> str:
    for key in ("date", "dates", "filename", "file", "hr_file", "hr_filename"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            return str(value[0])
        if torch.is_tensor(value):
            try:
                return str(value[0].item())
            except Exception:
                return str(value[0])
        return str(value)
    return fallback


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "sample"


def _tensor_to_numpy(t: torch.Tensor | None) -> np.ndarray:
    if t is None:
        return np.array([])
    return t.detach().cpu().numpy()


def _save_package(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    sample_id: str,
    dataset_index: int,
    member_index: int,
    batch: dict[str, Any],
    final_sample: torch.Tensor,
    capture_out: dict[str, torch.Tensor],
    cond_img: torch.Tensor | None,
    cond_local: torch.Tensor | None,
    lsm_cond: torch.Tensor | None,
    topo_cond: torch.Tensor | None,
    y: torch.Tensor | None,
    lr_ups: torch.Tensor | None,
    back_transforms: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _apply_back_transform(transform, value):
        if value is None:
            return np.array([])
        value_cpu = value.detach().cpu()
        if transform is None:
            return np.array([])
        try:
            out = transform(value_cpu)
        except TypeError:
            out = transform(value_cpu.numpy())
        if torch.is_tensor(out):
            return out.detach().cpu().numpy()
        return np.asarray(out)

    hr_var = cfg.get("highres", {}).get("variable", "prcp")
    bt_generated = back_transforms.get("generated") if back_transforms is not None else None
    bt_hr = back_transforms.get(f"{hr_var}_hr") if back_transforms is not None else None
    bt_lr = back_transforms.get(f"{hr_var}_lr") if back_transforms is not None else None
    physical_space_saved = bt_generated is not None

    arrays: dict[str, np.ndarray] = {}
    for key, value in capture_out.items():
        arrays[f"capture_{key}"] = _tensor_to_numpy(value)

    arrays["final_sample"] = _tensor_to_numpy(final_sample[0])
    if bt_generated is not None:
        arrays["final_sample_phys"] = _apply_back_transform(bt_generated, final_sample[0])
    if bt_generated is not None and "denoised" in capture_out:
        arrays["capture_denoised_phys"] = _apply_back_transform(bt_generated, capture_out["denoised"])
    if bt_generated is not None and "x" in capture_out:
        arrays["capture_x_phys"] = _apply_back_transform(bt_generated, capture_out["x"])
    if bt_lr is not None and cond_local is not None and cond_local.ndim == 4 and cond_local.shape[1] > 0:
        arrays["cond_local_lr_phys"] = _apply_back_transform(bt_lr, cond_local[:, 0:1])[0]
    arrays["cond_img_model_input"] = _tensor_to_numpy(cond_img[0] if cond_img is not None else None)
    arrays["cond_local_lr"] = _tensor_to_numpy(cond_local[0] if cond_local is not None else None)
    arrays["lsm_cond"] = _tensor_to_numpy(lsm_cond[0] if lsm_cond is not None else None)
    arrays["topo_cond"] = _tensor_to_numpy(topo_cond[0] if topo_cond is not None else None)
    arrays["y"] = _tensor_to_numpy(y[0] if y is not None else None)
    arrays["lr_ups"] = _tensor_to_numpy(lr_ups[0] if lr_ups is not None else None)

    hr_target_tensor = None
    for key in (
        "hr",
        "x",
        "target",
        "target_hr",
        hr_var,
        f"{hr_var}_hr",
        f"{hr_var}_target",
    ):
        if key in batch and batch[key] is not None:
            value = batch[key]
            if torch.is_tensor(value):
                hr_target_tensor = value[0]
                arrays["hr_target"] = _tensor_to_numpy(hr_target_tensor)
            else:
                arrays["hr_target"] = np.asarray(value[0])
            break
    if bt_hr is not None and hr_target_tensor is not None:
        arrays["hr_target_phys"] = _apply_back_transform(bt_hr, hr_target_tensor)

    np.savez_compressed(out_dir / "trajectory.npz", **arrays)

    metadata = {
        "sample_id": sample_id,
        "dataset_index": int(dataset_index),
        "member_index": int(member_index),
        "seed": int(args.seed + member_index),
        "config": str(args.config),
        "checkpoint": str(args.resolved_checkpoint),
        "split": str(args.split),
        "capture_every": int(args.capture_every),
        "capture_dtype": str(args.capture_dtype),
        "sampler": {
            "num_steps": int(args.sampling_steps),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "rho": float(args.rho),
            "S_churn": float(args.S_churn),
            "S_min": float(args.S_min),
            "S_max": float(args.S_max) if np.isfinite(args.S_max) else "inf",
            "S_noise": float(args.S_noise),
            "sigma_star": float(args.sigma_star),
            "sigma_star_mode": str(args.sigma_star_mode),
            "ramp_start_frac": float(args.ramp_start_frac),
            "ramp_end_frac": float(args.ramp_end_frac),
        },
        "physical_space_saved": bool(physical_space_saved),
        "physical_units_note": "*_phys arrays are back-transformed on LUMI using generation._build_back_transforms(cfg).",
        "note": "Illustrative EDM sampling trajectory for NoiseToRain. Not an evaluation diagnostic.",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract captured EDM trajectories for Noise-to-Rain.")
    parser.add_argument("--config", required=True, help="Path to the YAML config used by the model.")
    parser.add_argument("--output-dir", required=True, help="Output root for trajectory packages.")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of dataset samples/dates to export.")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset index to start from.")
    parser.add_argument("--members-per-sample", type=int, default=1, help="Number of stochastic trajectories per selected sample.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Override cfg['training']['batch_size'] before building dataloaders. "
            "Useful for trajectory extraction, where the YAML training batch size can make "
            "the generation loader expose too few batches for start-index based selection."
        ),
    )
    parser.add_argument("--seed", type=int, default=504, help="Base sampling seed.")
    parser.add_argument("--capture-every", type=int, default=1, help="Capture every k EDM steps.")
    parser.add_argument("--capture-dtype", choices=("float16", "float32"), default="float32")

    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--S-churn", type=float, default=None)
    parser.add_argument("--S-min", type=float, default=None)
    parser.add_argument("--S-max", type=float, default=None)
    parser.add_argument("--S-noise", type=float, default=None)
    parser.add_argument("--sigma-star", type=float, default=None)
    parser.add_argument("--sigma-star-mode", choices=("global", "late_ramp"), default=None)
    parser.add_argument("--ramp-start-frac", type=float, default=None)
    parser.add_argument("--ramp-end-frac", type=float, default=None)
    return parser.parse_args()


def _fill_sampler_args_from_cfg(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    edm_cfg = cfg.get("edm", {}) or {}
    full_cfg = cfg.get("full_gen_eval", {}) or {}
    gen_cfg = cfg.get("generation", {}) or {}
    sampler_cfg = cfg.get("sampler", {}) or {}
    sampler_grid = full_cfg.get("sampler_grid", {}) or {}
    sigma_ctrl = full_cfg.get("sigma_control", {}) or {}

    def first_grid(name: str, default: Any) -> Any:
        value = sampler_grid.get(name, default)
        if isinstance(value, list):
            return value[0]
        return value

    args.sampling_steps = int(
        args.sampling_steps
        if args.sampling_steps is not None
        else edm_cfg.get("sampling_steps", gen_cfg.get("sampler_steps", sampler_cfg.get("n_timesteps", 56)))
    )
    args.sigma_min = float(args.sigma_min if args.sigma_min is not None else edm_cfg.get("sigma_min", 0.002))
    args.sigma_max = float(args.sigma_max if args.sigma_max is not None else edm_cfg.get("sigma_max", 80.0))
    args.rho = float(args.rho if args.rho is not None else first_grid("rho", edm_cfg.get("rho", 7.0)))
    args.S_churn = float(args.S_churn if args.S_churn is not None else first_grid("S_churn", edm_cfg.get("S_churn", 0.0)))
    args.S_min = float(args.S_min if args.S_min is not None else edm_cfg.get("S_min", 0.0))
    args.S_max = float(args.S_max if args.S_max is not None else edm_cfg.get("S_max", float("inf")))
    args.S_noise = float(args.S_noise if args.S_noise is not None else edm_cfg.get("S_noise", 1.0))
    args.sigma_star = float(args.sigma_star if args.sigma_star is not None else edm_cfg.get("sigma_star", 1.0))
    args.sigma_star_mode = str(args.sigma_star_mode or sigma_ctrl.get("sigma_star_mode", "global"))
    args.ramp_start_frac = float(args.ramp_start_frac if args.ramp_start_frac is not None else sigma_ctrl.get("ramp_start_frac", 0.60))
    args.ramp_end_frac = float(args.ramp_end_frac if args.ramp_end_frac is not None else sigma_ctrl.get("ramp_end_frac", 0.85))
    return args


def main() -> None:
    _setup_logging()
    args = _parse_args()
    _set_seed(args.seed)
    device = _device()

    cfg = _load_yaml(args.config)
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = int(args.batch_size)
        logger.info("Overriding cfg['training']['batch_size'] for sonification extraction: %s", args.batch_size)

    # get_dataloader(cfg) constructs the generation loader from
    # cfg['data_handling']['n_gen_samples'], not from cfg['training']['batch_size'].
    # For start-index based trajectory extraction we therefore need the generation
    # subset to contain at least start_index + num_samples dates. Otherwise a config
    # with n_gen_samples=8 yields len(gen_loader.dataset)=8 and START_INDEX=120
    # skips every available sample.
    data_handling_cfg = cfg.setdefault("data_handling", {})
    requested_end_index = int(args.start_index + args.num_samples)
    original_n_gen_samples = int(data_handling_cfg.get("n_gen_samples", 0) or 0)
    if original_n_gen_samples < requested_end_index:
        data_handling_cfg["n_gen_samples"] = requested_end_index
        logger.info(
            "Expanding cfg['data_handling']['n_gen_samples'] for sonification extraction: %s -> %s "
            "so START_INDEX=%s and NUM_SAMPLES=%s are reachable.",
            original_n_gen_samples,
            requested_end_index,
            args.start_index,
            args.num_samples,
        )
    else:
        logger.info(
            "Keeping cfg['data_handling']['n_gen_samples']=%s for sonification extraction; requested_end_index=%s.",
            original_n_gen_samples,
            requested_end_index,
        )

    cfg.setdefault("runtime", {})
    cfg["runtime"].update(
        {
            "distributed": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main_process": True,
        }
    )
    cfg.setdefault("data_handling", {})["num_workers"] = int(cfg.get("data_handling", {}).get("num_workers", 0) or 0)
    cfg.setdefault("generation", {})["seed"] = int(args.seed)
    cfg.setdefault("full_gen_eval", {})["split"] = args.split
    args = _fill_sampler_args_from_cfg(args, cfg)

    logger.info("Building model with training_utils.get_model(cfg)")
    model, resolved_checkpoint = _load_model_from_cfg_checkpoint(cfg, device)
    args.resolved_checkpoint = resolved_checkpoint
    logger.info("Loaded config-derived checkpoint: %s", resolved_checkpoint)

    logger.info("Building dataloaders with training_utils.get_dataloader(cfg)")
    train_loader, val_loader, gen_loader = get_dataloader(cfg, verbose=True)
    loader_by_split = {"train": train_loader, "valid": val_loader, "test": gen_loader}
    loader = loader_by_split[args.split]
    try:
        loader_len = len(loader)
        dataset_len = len(loader.dataset)
        logger.info("Selected %s loader: len(loader)=%s | len(dataset)=%s", args.split, loader_len, dataset_len)
        logger.info(
            "Selection request: start_index=%s | num_samples=%s | requested_end_index=%s | members_per_sample=%s",
            args.start_index,
            args.num_samples,
            args.start_index + args.num_samples,
            args.members_per_sample,
        )
        if args.start_index >= dataset_len:
            logger.warning(
                "START_INDEX=%s is outside the selected loader dataset length=%s. "
                "No samples will be exported unless data_handling.n_gen_samples is increased.",
                args.start_index,
                dataset_len,
            )
        elif args.start_index + args.num_samples > dataset_len:
            logger.warning(
                "Requested samples extend past selected loader dataset length: start_index + num_samples = %s, dataset_len = %s. "
                "Only %s dataset samples can be exported.",
                args.start_index + args.num_samples,
                dataset_len,
                max(0, dataset_len - args.start_index),
            )
    except Exception:
        logger.info("Selected %s loader: could not inspect loader/dataset length", args.split)

    logger.info("Building back-transforms with generation._build_back_transforms(cfg)")
    back_transforms = _build_back_transforms(cfg)
    logger.info("Back-transform keys: %s", sorted(list(back_transforms.keys())))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    exported_samples = 0
    seen = 0
    for batch_idx, batch in enumerate(loader):
        batch_size = _infer_batch_size(batch)
        logger.info(
            "Processing loader batch %s: batch_size=%s | dataset_index range [%s, %s]",
            batch_idx,
            batch_size,
            seen,
            seen + batch_size - 1,
        )
        for k, v in batch.items():

            if torch.is_tensor(v):
                logger.info(
                    "[batch-debug] %s -> tensor shape=%s",
                    k,
                    tuple(v.shape),
                )

            elif isinstance(v, (list, tuple)):
                logger.info(
                    "[batch-debug] %s -> list len=%s",
                    k,
                    len(v),
                )

            else:
                logger.info(
                    "[batch-debug] %s -> type=%s",
                    k,
                    type(v),
                )
        for local_idx in range(batch_size):
            dataset_index = seen + local_idx
            if dataset_index < args.start_index:
                continue
            if exported_samples >= args.num_samples:
                break

            single = _extract_batch_item(batch, local_idx)
            sample_id = _find_sample_id(single, fallback=f"idx{dataset_index:06d}")
            safe_sample_id = _safe_name(sample_id)

            cond_img = _build_cond_img(single, cfg, model, device)
            cond_local = _build_local_cond_img(single, cfg, device)
            lsm_cond = _optional_tensor(single, ("lsm_cond", "lsm", "lsm_hr", "land_sea_mask"), device)
            topo_cond = _optional_tensor(single, ("topo_cond", "topo", "topography", "topo_hr"), device)
            y = _optional_tensor(single, ("y", "doy", "season", "month"), device)
            lr_ups = _build_lr_baseline(single, cfg, device)

            for member_idx in range(args.members_per_sample):
                member_seed = int(args.seed + member_idx)
                _set_seed(member_seed)
                logger.info(
                    "Sampling trajectory: dataset_index=%s sample_id=%s member=%s seed=%s",
                    dataset_index,
                    sample_id,
                    member_idx,
                    member_seed,
                )

                with torch.no_grad():
                    result = edm_sampler(
                        model,
                        batch_size=1,
                        num_steps=args.sampling_steps,
                        device=device,
                        img_size=int(_safe_get(cfg, "highres.data_size", [128, 128])[0]),
                        y=y,
                        cond_img=cond_img,
                        lsm_cond=lsm_cond,
                        topo_cond=topo_cond,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        rho=args.rho,
                        S_churn=args.S_churn,
                        S_min=args.S_min,
                        S_max=args.S_max,
                        S_noise=args.S_noise,
                        lr_ups=lr_ups,
                        cfg_guidance=cfg.get("classifier_free_guidance", None),
                        cfg_diagnostics={"per_batch_stats": False},
                        sigma_star=args.sigma_star,
                        sigma_star_mode=args.sigma_star_mode,
                        ramp_start_frac=args.ramp_start_frac,
                        ramp_end_frac=args.ramp_end_frac,
                        capture={
                            "enabled": True,
                            "every": args.capture_every,
                            "batch_index": 0,
                            "dtype": args.capture_dtype,
                            "sources": ("x_in", "denoised", "x_euler", "x"),
                        },
                    )

                final_sample, capture_out = result
                out_dir = output_root / f"idx{dataset_index:06d}_{safe_sample_id}" / f"member{member_idx:03d}_seed{member_seed}"
                _save_package(
                    out_dir,
                    args=args,
                    cfg=cfg,
                    sample_id=sample_id,
                    dataset_index=dataset_index,
                    member_index=member_idx,
                    batch=single,
                    final_sample=final_sample,
                    capture_out=capture_out,
                    cond_img=cond_img,
                    cond_local=cond_local,
                    lsm_cond=lsm_cond,
                    topo_cond=topo_cond,
                    y=y,
                    lr_ups=lr_ups,
                    back_transforms=back_transforms,
                )
                logger.info("Saved %s", out_dir)

            exported_samples += 1

        seen += batch_size
        if exported_samples >= args.num_samples:
            break

    logger.info("Done. Exported %d dataset samples to %s", exported_samples, output_root)


if __name__ == "__main__":
    main()