"""
    EDM-downscaling - generation runner (mirrors TrainingPipeline_general.generate_and_plot_samples)

    Outputs per-day npzs:
        - ensembles/{date}.npz   -> {'ens': [M,1,H,W]}  (model space)
        - pmm/{date}.npz         -> {'pmm': [1,1,H,W]}  (model space)
        - lr_hr/{date}.npz       -> {'hr': [1,1,H,W] or None, 'lr_hr': [1,1,H,W] or None} (model space)
    and meta/manifest.json for reproducibility
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from sbgm.special_transforms import build_back_transforms_from_stats, lr_baseline_to_hr_zspace
from sbgm.utils import extract_samples, get_model_string, crop_bounds_to_stats_str


# --- Helper for resolving stats regions for transforms ---
def _resolve_generation_stats_regions(cfg: dict):
    """Resolve effective HR/LR stats regions for generation.

    Mirrors the Paper2 spatial-context conventions used in training/data loading:
      - large_domain: LR stats use the full LR domain, serialized in xxyy order.
      - colocated:   LR stats follow the HR/co-located cutout convention.
      - otherwise:   LR stats use lowres.cutout_domains in xxyy order.

    Returns a dict with domain strings, effective bounds, and serialized crop strings.
    """
    hr_full = cfg['highres'].get('full_domain_dims', None)
    lr_full = cfg['lowres'].get('full_domain_dims', None)

    hr_domain_str = f"{hr_full[0]}x{hr_full[1]}" if hr_full is not None else "full_domain"
    lr_domain_str = f"{lr_full[0]}x{lr_full[1]}" if lr_full is not None else "full_domain"

    hr_bounds = cfg['highres'].get('cutout_domains', None)
    lr_bounds_cfg = cfg['lowres'].get('cutout_domains', None)

    paper2 = (cfg.get('paper2', {}) or {})
    spatial = (paper2.get('spatial_context', {}) or {})
    spatial_mode = str(spatial.get('mode', '')).lower()

    hr_crop_str = crop_bounds_to_stats_str(hr_bounds, order="yyxx") if hr_bounds is not None else 'no_crop'

    if spatial_mode == 'large_domain' and (lr_full is not None):
        # Internal effective LR bounds are [x1, x2, y1, y2]
        lr_bounds_eff = [0, lr_full[1], 0, lr_full[0]]
        lr_crop_str = crop_bounds_to_stats_str(lr_bounds_eff, order="xxyy")
    elif spatial_mode == 'colocated':
        # Co-located mode should use the HR-aligned crop convention
        lr_bounds_eff = hr_bounds
        lr_crop_str = crop_bounds_to_stats_str(hr_bounds, order="yyxx") if hr_bounds is not None else 'no_crop'
    else:
        # Legacy/non-paper2 path: config cutout_domains are stored as [y1, y2, x1, x2]
        lr_bounds_eff = lr_bounds_cfg
        lr_crop_str = crop_bounds_to_stats_str(lr_bounds_eff, order="yyxx") if lr_bounds_eff is not None else 'no_crop'
    return {
        'spatial_mode': spatial_mode,
        'hr_domain_str': hr_domain_str,
        'lr_domain_str': lr_domain_str,
        'hr_bounds_eff': hr_bounds,
        'lr_bounds_eff': lr_bounds_eff,
        'hr_crop_str': hr_crop_str,
        'lr_crop_str': lr_crop_str,
    }
from sbgm.score_sampling import edm_sampler
from sbgm.monitoring import (
    report_precip_extremes,
)
from sbgm.evaluate_sbgm.metrics_univariate import (
    pmm_from_ensemble,
)

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    output_root: str
    ensemble_size: int = 32
    max_dates: Optional[int] = -1  # -1 means all dates
    sampler_steps: int = 40
    seed: int = 504

    # EDM controls
    use_edm: bool = True
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    S_churn: float = 0.0
    S_min: float = 0.0
    S_max: float = float("inf")
    S_noise: float = 1.0
    predict_residual: bool = False  # if True, pass lr_ups to sampler

    # Saving: "physical" (default), "model", or "both"
    save_space: str = "physical"
    physical_dtype: str = "float32"  # when saving physical space, cast to this dtype

    # Logging
    log_every: int = 25


def _save_npz(path: Path, **arrays):
    """Save arrays to npz at path, creating parent dirs if needed.

    Important: do NOT write None values into npz (they become dtype=object arrays).
    Also ensure torch.Tensors are converted to numpy arrays.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        if v is None:
            # Skip None to avoid dtype=object arrays inside the npz
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = np.asarray(v)

    np.savez_compressed(path, **out)

def _repeat_to_M(x, M: int):
    """ Prepare tensor to ensemble size M by repeating along batch dim."""
    if x is None:
        return None
    # Assume batch-first [B, ...]; take first item in batch and repeat to M
    x1 = x[:1]
    reps = [M] + [1] * (x1.dim() - 1)
    return x1.repeat(*reps)

def _build_back_transforms(cfg: dict):
    stats_regions = _resolve_generation_stats_regions(cfg)

    full_domain_dims_str_hr = stats_regions['hr_domain_str']
    crop_region_hr_str = stats_regions['hr_crop_str']
    full_domain_dims_str_lr = stats_regions['lr_domain_str']
    crop_region_lr_str = stats_regions['lr_crop_str']

    logger.info("[generation] _build_back_transforms spatial_mode: %s", stats_regions['spatial_mode'])
    logger.info("[generation] _build_back_transforms HR stats crop string: %s", crop_region_hr_str)
    logger.info("[generation] _build_back_transforms LR stats crop string: %s", crop_region_lr_str)

    return build_back_transforms_from_stats(
        hr_var=cfg['highres']['variable'],
        hr_model=cfg['highres']['model'],
        domain_str_hr=full_domain_dims_str_hr,
        crop_region_str_hr=crop_region_hr_str,
        hr_scaling_method=cfg['highres']['scaling_method'],
        hr_buffer_frac=cfg['highres'].get('buffer_frac', 0.0),
        lr_vars=cfg['lowres']['condition_variables'],
        lr_model=cfg['lowres']['model'],
        domain_str_lr=full_domain_dims_str_lr,
        crop_region_str_lr=crop_region_lr_str,
        lr_scaling_methods=cfg['lowres']['scaling_methods'],
        lr_buffer_frac=cfg['lowres'].get('buffer_frac', 0.0),
        split='train',  # match training.generate_and_plot_samples
        stats_dir_root=cfg['paths']['stats_load_dir'],
        eps=cfg['transforms'].get('prcp_eps', 0.01),
    )


class GenerationRunner:
    def __init__(self, model: torch.nn.Module, cfg: dict, device: str, out_root: Path, gen_config: GenerationConfig, quicklook: bool = False):

        self.model = model
        self.cfg = cfg
        self.device = device
        self.out_root = out_root
        # Ensure base output root exists even for quicklook (no subdirs yet=> 
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.gen_config = gen_config

        self.global_prcp_eps = cfg['transforms'].get('prcp_eps', 0.01)


        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.full_domain_dims_hr = cfg['highres']['full_domain_dims']
        self.crop_region_hr = cfg['highres']['cutout_domains']

        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']
        self.full_domain_dims_lr = cfg['lowres']['full_domain_dims']
        self.crop_region_lr = cfg['lowres']['cutout_domains']

        # Indices of LR channels corresponding to the HR target variable (can be multiple)
        self._lr_target_indices = []
        if self.hr_var in self.lr_vars:
            self._lr_target_indices = [i for i, v in enumerate(self.lr_vars) if v == self.hr_var]
            
        # Cache strings for stats lookups
        self._dom_hr_str = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        self._dom_lr_str = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        self._crop_hr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx") if self.crop_region_hr is not None else "no_crop"

        stats_regions = _resolve_generation_stats_regions(cfg)
        self._spatial_mode = stats_regions['spatial_mode']
        self._crop_hr_bounds_eff = stats_regions['hr_bounds_eff']
        self._crop_lr_bounds_eff = stats_regions['lr_bounds_eff']
        self._crop_hr_str = stats_regions['hr_crop_str']
        self._crop_lr_str = stats_regions['lr_crop_str']
        self._stats_root = self.cfg['paths']['stats_load_dir']

        logger.info("[generation] spatial_mode: %s", self._spatial_mode)
        logger.info("[generation] Cached HR stats crop string: %s", self._crop_hr_str)
        logger.info("[generation] Cached LR stats crop string: %s", self._crop_lr_str)
        self._hr_method_for_target = self.hr_scaling_method
        # Assume LR scaling methods is a list aligned with lr_vars; get method for the target variable
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self._lr_method_for_target = self.lr_scaling_methods[idx_t]
        else:
            self._lr_method_for_target = None  # Target variable not in LR vars
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot determine LR scaling method for target - residuals may not be aligned.")

        # cutout stationarity control + optional static LSM for evaluation
        # Support both plain dict configs and OmegaConf-style DictConfig.
        if isinstance(cfg, dict):
            eval_cfg = cfg.get("evaluation", None)
        else:
            eval_cfg = getattr(cfg, "evaluation", None)

        if isinstance(eval_cfg, dict):
            eval_stationary_cfg = eval_cfg.get("stationary_cutout", None)
        else:
            eval_stationary_cfg = getattr(eval_cfg, "stationary_cutout", None) if eval_cfg is not None else None

        if isinstance(eval_stationary_cfg, dict):
            # mirror training_utils logic: 'hr_enabled' and 'hr_bounds'
            self.stationary_cutout = bool(eval_stationary_cfg.get("hr_enabled", True))
            self._hr_bounds_eval = eval_stationary_cfg.get("hr_bounds", None)
        elif eval_stationary_cfg is not None:
            # allow legacy boolean / object configs and try to read hr_bounds attribute if present
            self.stationary_cutout = bool(eval_stationary_cfg)
            self._hr_bounds_eval = getattr(eval_stationary_cfg, "hr_bounds", None)
        else:
            # no explicit evaluation cutout config
            self.stationary_cutout = True            
            self._hr_bounds_eval = None

        # Fallbacks: allow geometry configured under full_gen_eval or highres if not set above
        if self._hr_bounds_eval is None:
            if isinstance(cfg, dict):
                fg = cfg.get("full_gen_eval", None)
            else:
                fg = getattr(cfg, "full_gen_eval", None)

            if isinstance(fg, dict):
                sc = fg.get("stationary_cutout", None)
            else:
                sc = getattr(fg, "stationary_cutout", None) if fg is not None else None

            hb = None
            if isinstance(sc, dict):
                hb = sc.get("hr_bounds", None)
            elif sc is not None:
                hb = getattr(sc, "hr_bounds", None)
            if hb is not None:
                self._hr_bounds_eval = hb

        if self._hr_bounds_eval is None:
            if isinstance(cfg, dict):
                highres_cfg = cfg.get("highres", None)
            else:
                highres_cfg = getattr(cfg, "highres", None)

            if isinstance(highres_cfg, dict):
                sc = highres_cfg.get("stationary_cutout", None)
            else:
                sc = getattr(highres_cfg, "stationary_cutout", None) if highres_cfg is not None else None

            hb = None
            if isinstance(sc, dict):
                hb = sc.get("bounds", None)
            elif sc is not None:
                hb = getattr(sc, "bounds", None)
            if hb is not None:
                self._hr_bounds_eval = hb


        logger.info("[generation] Evaluation HR bounds for static LSM: %s", self._hr_bounds_eval)

        self._first_lsm: Optional[torch.Tensor] = None
        self._lsm_stationary_ok: bool = True
        self._lsm_static: Optional[torch.Tensor] = None

        # dirs (if quicklook, don't create)
        if not quicklook:
            (self.out_root / 'lsm').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'ensembles').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'pmm').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'lr_hr').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'meta').mkdir(parents=True, exist_ok=True)

            # Additional dirs when saving physical/model explicitly
            if self.gen_config.save_space in ('physical', 'both'):
                (self.out_root / 'ensembles_phys').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'pmm_phys').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'lr_hr_phys').mkdir(parents=True, exist_ok=True)
            if self.gen_config.save_space in ('model', 'both'):
                (self.out_root / 'ensembles_model').mkdir(parents=True, exist_ok=True)
                (self.out_root / 'pmm_model').mkdir(parents=True, exist_ok=True)

        # sentinel 
        mon_cfg = cfg.get('monitoring', {}).get('extreme_prcp', {})
        self.sentinel_thr = float(mon_cfg.get('threshold_mm', 500.0)) # mm/day
        self.clamp_in_gen = bool(mon_cfg.get('clamp_in_generation', True))

        # back-transforms for quick sanity checks
        self.back_transforms = _build_back_transforms(cfg)
        self.bt_gen = self.back_transforms.get('generated', None)
        self.bt_hr = self.back_transforms.get(f"{cfg['highres']['variable']}_hr", None)
        # Get the matching LR back-transform for the target variable, if available
        self.bt_lr_target = None
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self.bt_lr_target = self.back_transforms.get(f"{self.hr_var}_lr", None)
        else:
            self.bt_lr_target = None
            logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot build LR back-transform for target variable.")
        
        self.bt_lr_lrspace = self.back_transforms.get(f"{self.hr_var}_lr_lrspace") or self.back_transforms.get(f"{self.hr_var}_lr")
        self.bt_lr_hrspace = self.back_transforms.get(f"{self.hr_var}_lr_hrspace")

        # sampler selection
        if bool(cfg.get('edm', {}).get('enabled', False)) or self.gen_config.use_edm:
            self._sampler_kind = 'edm'
            self._sampler_fn = edm_sampler
        else:
            logger.error("Currently only EDM sampler is supported in generation.")
            # st = cfg.get('sampler', {}).get('sampler_type', 'pc_sampler')
            # if st == 'pc_sampler':
            #     self._sampler_kind = 'sde'
            #     self._sampler_fn = pc_sampler
            # elif st == 'Euler_Maruyama_sampler':
            #     self._sampler_kind = 'sde'
            #     self._sampler_fn = Euler_Maruyama_sampler
            # elif st == 'ode_sampler':
            #     self._sampler_kind = 'ode'
            #     self._sampler_fn = ode_sampler
            # else:
            #     raise ValueError(f"Unknown sampler_type {st} in config")

    def _build_cond_img(self, batch: dict) -> torch.Tensor | None:
        """Build the conditioning image passed to the UNet encoder.

        Mirrors TrainingPipeline_general._build_cond_img, including Paper2 context encoder support.

        Returns:
            - local-only conditioning if context encoder disabled
            - local+context (or context-only) if enabled
        """
        if self.lr_vars is None or len(self.lr_vars) == 0:
            return None

        paper2 = (self.cfg.get("paper2", {}) or {})
        spatial = (paper2.get("spatial_context", {}) or {})
        mode = str(spatial.get("mode", "")).lower()

        lr_tensors_local = []
        for v in self.lr_vars:
            k_local = f"{v}_lr_local"
            k_ctx = f"{v}_lr"

            if mode == "large_domain":
                if (k_local not in batch) or (batch[k_local] is None):
                    raise KeyError(
                        f"paper2.spatial_context.mode='large_domain' requires '{k_local}' "
                        f"(got keys: {list(batch.keys())})."
                    )
                lr_tensors_local.append(batch[k_local])
            else:
                if (k_local in batch) and (batch[k_local] is not None):
                    lr_tensors_local.append(batch[k_local])
                else:
                    lr_tensors_local.append(batch[k_ctx])

        cond_local = torch.cat(lr_tensors_local, dim=1).to(self.device)

        enc_cfg = (spatial.get("encoder", {}) or {})
        use_ctx = bool(enc_cfg.get("enabled", False)) and (getattr(self.model, "context_encoder", None) is not None)
        ctx_mode = str(enc_cfg.get("input_mode", "context_plus_local"))

        if not use_ctx:
            return cond_local

        xs = []
        for v in self.lr_vars:
            k = f"{v}_lr"
            t = batch[k]  # [B,1,H_lr,W_lr] (or [B,2,...] for dual_lr)
            xs.append(t[:, 0])
        x_bvhw = torch.stack(xs, dim=1)              # [B,V,H_lr,W_lr]
        x_ctx = x_bvhw.unsqueeze(1).to(self.device)  # [B,1,V,H_lr,W_lr]

        ctx = self.model.encode_spatial_context(x_ctx)  # [B,Cctx,128,128]

        if ctx_mode == "context_only":
            return ctx

        return torch.cat([cond_local, ctx], dim=1)
    
    def _get_static_lsm(self) -> Optional[torch.Tensor]:
        """
        Fallback land-sea mask used when the dataset does not yield an LSM tensor.

        This loads the full-domain LSM from cfg['paths']['lsm_path'], flips it to match
        the DANRA orientation (np.flipud), and then crops to the evaluation HR bounds
        if they are set. The HR bounds are resolved in this order:
          1) cfg['evaluation']['stationary_cutout']['hr_bounds']
          2) cfg['full_gen_eval']['stationary_cutout']['hr_bounds']
          3) cfg['highres']['stationary_cutout']['bounds']

        The result is cached as a torch.bool tensor on CPU.
        """
        # Return cached static mask if already built
        if self._lsm_static is not None:
            return self._lsm_static

        # Resolve LSM path
        try:
            lsm_path = self.cfg["paths"]["lsm_path"]
        except Exception as e:
            logger.warning("[generation] _get_static_lsm: could not resolve paths.lsm_path: %s", e)
            return None

        # Load LSM from npz or raw array
        try:
            arr = np.load(lsm_path, allow_pickle=True)
            if isinstance(arr, np.lib.npyio.NpzFile):  # type: ignore
                data = None
                # Try typical keys in order of preference
                for k in ("lsm_hr", "lsm", "data", "mask"):
                    if k in arr.files:
                        data = arr[k]
                        break
                if data is None:
                    logger.warning(
                        "[generation] _get_static_lsm: no suitable key in %s (tried lsm_hr, lsm, data, mask)",
                        lsm_path,
                    )
                    return None
            else:
                data = arr
        except Exception as e:
            logger.warning("[generation] _get_static_lsm: failed to load LSM from %s: %s", lsm_path, e)
            return None

        data = np.asarray(data)
        # Ensure 2D; squeeze singleton dimensions if necessary
        if data.ndim != 2:
            data = np.squeeze(data)
            if data.ndim != 2:
                logger.warning(
                    "[generation] _get_static_lsm: expected 2D LSM array, got shape %s after squeeze", data.shape
                )
                return None

        # Flip to match DANRA orientation (as in previous implementation)
        data = np.flipud(data)

        # Optional crop to stationary HR evaluation bounds [y0, y1, x0, x1]
        hr_bounds = self._hr_bounds_eval
        if hr_bounds is not None:
            if len(hr_bounds) != 4:
                logger.warning(
                    "[generation] _get_static_lsm: hr_bounds must have length 4, got %s", hr_bounds
                )
            else:
                y0, y1, x0, x1 = [int(v) for v in hr_bounds]
                data = data[y0:y1, x0:x1]

        # Convert to bool mask and cache
        lsm_bool = (data > 0.5)
        self._lsm_static = torch.from_numpy(lsm_bool.astype(np.bool_))
        logger.info(
            "[generation] Built static LSM from %s with shape %s (hr_bounds=%s)",
            lsm_path,
            tuple(self._lsm_static.shape),
            hr_bounds,
        )

        return self._lsm_static
    
    @torch.no_grad()
    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None) -> torch.Tensor:
        """Build the LR upsampled baseline used for residual prediction.

        This MUST be aligned with the dataset channel layout (see `_build_cond_channel_map` in `data_modules.py`)
        and with the logic used in `run()` when extracting/saving LR channels.

        Returns:
            Tensor [B,1,H,W] in the requested baseline_space:
              - baseline_space='lr'   -> LR-space (normalized with LR stats)
              - baseline_space='hr'   -> HR-space (normalized with HR stats)
              - baseline_space='auto' -> prefer HR-space if available, else convert LR->HR
        """
        if cond_images is None:
            raise ValueError("cond_images is None, cannot extract LR baseline for residual prediction.")

        # Config
        target_var = self.hr_var
        cond_vars = list(self.cfg.get('lowres', {}).get('condition_variables', []))
        if target_var not in cond_vars:
            raise ValueError(
                f"Target variable '{target_var}' not found in condition variables {cond_vars}, "
                "cannot extract LR baseline for residual prediction."
            )

        dual_lr = bool(self.cfg.get('lowres', {}).get('dual_lr', False))
        main_scale = str(self.cfg.get('lowres', {}).get('lr_main_var_scale', 'LR')).upper()
        baseline_space = str(self.cfg.get('edm', {}).get('baseline_space', 'hr')).lower()
        
        # Reconstruct the expected channel indices deterministically (mirrors `_build_cond_channel_map`)
        c = 0
        target_main_idx = None
        target_lronly_idx = None
        for cond in cond_vars:
            if dual_lr and (cond == target_var):
                target_main_idx = c
                target_lronly_idx = c + 1
                c += 2
            else:
                if cond == target_var:
                    target_main_idx = c
                c += 1

        if target_main_idx is None:
            raise ValueError(
                f"Could not determine baseline channel index for target '{target_var}'. "
                f"cond_vars={cond_vars}, dual_lr={dual_lr}"
            )
        if cond_images.shape[1] <= target_main_idx:
            raise ValueError(
                f"cond_images has shape {tuple(cond_images.shape)}, but needs channel index {target_main_idx} "
                f"for target '{target_var}'."
            )
        
        # Slice candidate channels
        main_chan = cond_images[:, target_main_idx:target_main_idx+1, :, :]  # [B,1,H,W]
        lronly_chan = None
        if target_lronly_idx is not None and cond_images.shape[1] > target_lronly_idx:
            lronly_chan = cond_images[:, target_lronly_idx:target_lronly_idx+1, :, :]
        # Determine what "space" the main channel lives in
        # - if lr_main_var_scale == 'LR' -> main is LR-space
        # - else ('HR', 'HR_LR', ...)    -> main is HR-space
        main_is_lrspace = (main_scale == 'LR')

        # Pick an LR-space candidate (needed for baseline_space='lr' and for LR->HR conversion)
        lrspace_chan = None
        if lronly_chan is not None:
            # In dual-LR, the lr_only channel is always LR-space by definition
            lrspace_chan = lronly_chan
        elif main_is_lrspace:
            lrspace_chan = main_chan

        # Pick an HR-space candidate (needed for baseline_space='hr'/'auto' when available)
        hrspace_chan = None
        if (not main_is_lrspace):
            hrspace_chan = main_chan

        # baseline_space == 'lr' -> must return LR-space (no conversion)
        if baseline_space == 'lr':
            if lrspace_chan is None:
                raise ValueError(
                    f"baseline_space='lr' but no LR-space baseline channel exists for target '{target_var}'. "
                    f"dual_lr={dual_lr}, lr_main_var_scale={main_scale}."
                )
            logger.info("[generation] lr_ups baseline: using LR-space channel (baseline_space='lr').")
            return lrspace_chan

        # baseline_space == 'hr' or 'auto'
        if baseline_space in ('hr', 'auto'):
            if hrspace_chan is not None:
                # Already in HR-space
                logger.info("[generation] lr_ups baseline: using HR-space channel (already HR-space).")
                return hrspace_chan

            # Need to convert LR-space -> HR-space
            if lrspace_chan is None:
                raise ValueError(
                    f"baseline_space='{baseline_space}' requires HR-space baseline, but neither an HR-space channel "
                    f"nor an LR-space channel is available for target '{target_var}'."
                )

            lr_method_for_baseline = self._lr_method_for_target
            if lr_method_for_baseline is None:
                raise ValueError(
                    "LR scaling method for baseline is None. Cannot proceed with lr_baseline_to_hr_zspace. "
                    "Please check your configuration."
                )

            logger.info(
                "[generation] LR->HR baseline conversion stats: spatial_mode=%s | lr_domain=%s | lr_crop=%s | hr_domain=%s | hr_crop=%s",
                getattr(self, '_spatial_mode', 'unknown'),
                self._dom_lr_str,
                self._crop_lr_str,
                self._dom_hr_str,
                self._crop_hr_str,
            )
            lr_in_hr_space = lr_baseline_to_hr_zspace(
                lr_chan_norm=lrspace_chan,
                # LR meta
                lr_variable=self.hr_var,
                lr_model=self.cfg['lowres']['model'],
                lr_domain_str=self._dom_lr_str,
                lr_crop_region_str=self._crop_lr_str,
                lr_split=self.cfg['transforms'].get('scaling_split', 'train'),
                lr_scaling_method=lr_method_for_baseline,
                lr_buffer_frac=self.cfg['lowres'].get('buffer_frac', 0.0),
                lr_stats_dir_root=self.cfg['paths']['stats_load_dir'],
                # HR meta
                hr_variable=self.hr_var,
                hr_model=self.cfg['highres']['model'],
                hr_domain_str=self._dom_hr_str,
                hr_crop_region_str=self._crop_hr_str,
                hr_split=self.cfg['transforms'].get('scaling_split', 'train'),
                hr_scaling_method=self.hr_scaling_method,
                hr_buffer_frac=self.cfg['highres'].get('buffer_frac', 0.0),
                hr_stats_dir_root=self.cfg['paths']['stats_load_dir'],
                eps=self.global_prcp_eps,
            )

            return lr_in_hr_space

        raise ValueError(f"Unknown edm.baseline_space='{baseline_space}'. Expected one of: 'lr', 'hr', 'auto'.")


    @torch.no_grad()
    def run(self, gen_dataloader, save=True, output_results: bool = False):
        model_name = get_model_string(self.cfg)
        logger.info(f"[generation] Using model: {model_name}")

        n_days = 0
        edm_cfg = self.cfg.get('edm', {})
        guidance_cfg = self.cfg.get('classifier_free_guidance', {})
        M = int(self.gen_config.ensemble_size)
        steps = int(edm_cfg.get('sampling_steps', self.gen_config.sampler_steps)) # default to config value if not in model cfg
        img_size = int(self.cfg['highres']['data_size'][0])  # assume square 
        results = [] if output_results else None

        logger.info(f"[generation] cfg: ensemble_size={M}, steps={steps}, save_space={self.gen_config.save_space}, predict_residual={self.gen_config.predict_residual}, out_root={self.out_root}")

        for idx, samples in enumerate(tqdm(gen_dataloader, desc="Generating samples", unit='batch')):
            hr_phys = None
            lr_phys = None
            dates = samples['date'] if ('date' in samples and isinstance(samples['date'], (list, tuple)) and len(samples['date']) > 0) else [f"idx{idx:04d}"]
            date0 = dates[0]  # use first date for naming
            logger.info(f"[generation] Generating for date {date0} ({idx+1}/{len(gen_dataloader)}) with ensemble size {M}.")
            # Extract model-space tensors
            x_gen, y_gen, cond_images_gen, lsm_hr_gen, lsm_gen, sdf_gen, topo_gen, hr_points_gen, lr_points_gen = extract_samples(samples, self.device)
            # ------------------------------------------------------------
            # Build conditioning exactly like training (Paper2 local + context).
            # Keep a *local-only* copy for lr_ups_baseline when predict_residual=True.
            # ------------------------------------------------------------
            local_cond = None
            try:
                lr_vars = list((self.cfg.get('lowres', {}) or {}).get('condition_variables', []) or [])
                if len(lr_vars) > 0:
                    local_list = []
                    for v in lr_vars:
                        k_local_v = f"{v}_lr_local"
                        k_large_v = f"{v}_lr"
                        if isinstance(samples, dict) and (k_local_v in samples) and (samples[k_local_v] is not None):
                            t = samples[k_local_v]
                        elif isinstance(samples, dict) and (k_large_v in samples) and (samples[k_large_v] is not None):
                            t = samples[k_large_v]
                        else:
                            t = None

                        if t is not None:
                            if not torch.is_tensor(t):
                                t = torch.tensor(t)
                            t = t.to(self.device)
                            if t.ndim == 3:
                                t = t.unsqueeze(1)
                            elif t.ndim == 2:
                                t = t.unsqueeze(0).unsqueeze(0)
                            # enforce 1-channel per var for local stack
                            if t.ndim == 4 and t.shape[1] > 1:
                                t = t[:, :1]
                            local_list.append(t)

                    if len(local_list) > 0:
                        local_cond = torch.cat(local_list, dim=1)
            except Exception as e:
                if idx == 0:
                    logger.warning("[generation] Failed to rebuild local_cond from batch keys; will fall back to extracted cond_images. err=%s", e)

            # Build cond_images for the UNet (may include context encoder channels)
            try:
                cond_images_gen = self._build_cond_img(samples)
            except Exception as e:
                if idx == 0:
                    logger.warning("[generation] _build_cond_img failed; using extracted cond_images. err=%s", e)

            # Fallback if local_cond couldn't be rebuilt
            if local_cond is None:
                local_cond = cond_images_gen

            # Always feed HR co-located statics (match training)
            lsm_cond = lsm_hr_gen if lsm_hr_gen is not None else lsm_gen
            topo_hr_gen = None
            if isinstance(samples, dict) and ("topo_hr" in samples) and (samples["topo_hr"] is not None):
                try:
                    topo_hr_gen = samples["topo_hr"].to(self.device)
                except Exception:
                    topo_hr_gen = None
            topo_cond = topo_hr_gen if topo_hr_gen is not None else topo_gen

            if idx == 0:
                try:
                    logger.info(
                        "[generation][cond] cond_images_gen=%s, local_cond=%s, lsm_cond=%s, topo_cond=%s",
                        None if cond_images_gen is None else tuple(cond_images_gen.shape),
                        None if local_cond is None else tuple(local_cond.shape),
                        None if lsm_cond is None else tuple(lsm_cond.shape),
                        None if topo_cond is None else tuple(topo_cond.shape),
                    )
                except Exception:
                    pass

            # Align LR 
            k_local = f"{self.hr_var}_lr_local"
            k_large = f"{self.hr_var}_lr"

            lr_local = samples.get(k_local, None)   # expected [B,1,128,128]
            lr_large = samples.get(k_large, None)   # expected [B,1,589,789] (Paper2 large domain)

            if lr_local is not None:
                lr_local = lr_local.to(self.device)
                if lr_local.ndim == 3: lr_local = lr_local.unsqueeze(1)
                if lr_local.ndim == 4 and lr_local.shape[1] != 1: lr_local = lr_local[:, :1]

            if lr_large is not None:
                lr_large = lr_large.to(self.device)
                if lr_large.ndim == 3: lr_large = lr_large.unsqueeze(1)
                if lr_large.ndim == 4 and lr_large.shape[1] != 1: lr_large = lr_large[:, :1]

            # --- Save/check land-sea mask(s) ---
            try:
                # Prefer LSM from dataset; if missing, fall back to static LSM from paths.lsm_path
                lsm0 = None

                # lsm_hr_gen is expected as [B,1,H,W] bool/0-1
                lsm = lsm_hr_gen
                if lsm is not None and torch.is_tensor(lsm):
                    lsm_cpu = (lsm.detach().cpu() > 0.5).to(torch.bool)
                    # assume B==1 in generation; take [0]
                    lsm0 = lsm_cpu[0, 0] if lsm_cpu.dim() == 4 else lsm_cpu.squeeze()
                else:
                    static_lsm = self._get_static_lsm()
                    if static_lsm is not None:
                        lsm0 = static_lsm
                        logger.info("[generation] Using static LSM fallback for date %s", dates[0])

                if lsm0 is not None:                    
                    # Always save per-date mask if saving is enabled
                    if save:
                        _save_npz(self.out_root / 'lsm' / f'{dates[0]}.npz', lsm_hr=lsm0.numpy())
                        logger.info("[generation] Saved per-date land mask → %s", self.out_root / 'lsm' / f'{dates[0]}.npz')
                    # Set/compare canonical mask, and save canonical on first encounter if saving
                    if self._first_lsm is None:
                        self._first_lsm = lsm0.clone()
                        if self.stationary_cutout and save:
                            _save_npz(self.out_root / 'meta' / 'land_mask.npz', lsm_hr=lsm0.numpy())
                            logger.info("[generation] Saved canonical land mask → %s", self.out_root / 'meta' / 'land_mask.npz')
                    else:
                        if not torch.equal(self._first_lsm, lsm0):
                            self._lsm_stationary_ok = False
            except Exception as e:
                logger.warning(f"[generation] Could not record LSM for {dates[0]}: {e}")

            # Optional baseline for residual EDM
            lr_ups_baseline = None
            if (bool(self.cfg.get('edm', {}).get('enabled', False)) and bool(self.cfg['edm'].get('predict_residual', False))) or self.gen_config.predict_residual:
                if local_cond is None:
                    raise ValueError("predict_residual=True but local_cond is None; cannot build lr_ups baseline.")
                lr_ups_baseline = self._build_lr_ups_baseline(local_cond).to(self.device)

                # Convert LR baseline into HR z-space if needed (mirrors training intent)
                try:
                    logger.info(
                        "[generation] run(): LR->HR baseline conversion stats: spatial_mode=%s | lr_domain=%s | lr_crop=%s | hr_domain=%s | hr_crop=%s",
                        getattr(self, '_spatial_mode', 'unknown'),
                        self._dom_lr_str,
                        self._crop_lr_str,
                        self._dom_hr_str,
                        self._crop_hr_str,
                    )
                    lr_ups_baseline = lr_baseline_to_hr_zspace(
                        lr_local,
                        hr_scaling_method=self._hr_method_for_target,
                        lr_scaling_method=self._lr_method_for_target,
                        stats_dir_root=self._stats_root,
                        hr_model=self.cfg['highres']['model'],
                        lr_model=self.cfg['lowres']['model'],
                        var=self.hr_var,
                        domain_str_hr=self._dom_hr_str,
                        crop_region_str_hr=self._crop_hr_str,
                        domain_str_lr=self._dom_lr_str,
                        crop_region_str_lr=self._crop_lr_str,
                        split='train',
                        eps=self.cfg['transforms'].get('prcp_eps', 0.01),
                    )
                except TypeError:
                    # if your helper has an older signature in this repo state
                    lr_ups_baseline = lr_baseline_to_hr_zspace(lr_local, self.back_transforms)

                lr_ups_baseline = lr_ups_baseline.to(self.device)

            # Freeze conditioning to a single date and tile to M samples
            if x_gen is not None and x_gen.shape[0] != 1:
                logger.warning(f"[generation] x_gen batch size {x_gen.shape[0]} != 1; freezing to first item and tiling to ensemble size {M}.")


            y_1 = y_gen[:1] if y_gen is not None else None
            cond_img_1 = cond_images_gen[:1] if cond_images_gen is not None else None
            lsm_1 = lsm_cond[:1] if lsm_cond is not None else None
            topo_1 = topo_cond[:1] if topo_cond is not None else None
            lr_ups_1 = lr_ups_baseline[:1] if lr_ups_baseline is not None else None
            x_hr_1 = x_gen[:1] if x_gen is not None else None

            y_M = _repeat_to_M(y_1, M)
            cond_images_M = _repeat_to_M(cond_img_1, M)
            lsm_M = _repeat_to_M(lsm_1, M)
            topo_M = _repeat_to_M(topo_1, M)
            lr_ups_M = _repeat_to_M(lr_ups_1, M)

            # Sample ensemble (model space)
            if self._sampler_kind == 'edm':
                generated = self._sampler_fn(
                    score_model=self.model,
                    batch_size=M,
                    num_steps=steps,
                    device=self.device,
                    img_size=img_size,
                    y=y_M,
                    cond_img=cond_images_M,
                    lsm_cond=lsm_M,
                    topo_cond=topo_M,
                    sigma_min=float(edm_cfg.get('sigma_min', self.gen_config.sigma_min)),
                    sigma_max=float(edm_cfg.get('sigma_max', self.gen_config.sigma_max)),
                    rho=float(edm_cfg.get('rho', self.gen_config.rho)),
                    S_churn=float(edm_cfg.get('S_churn', self.gen_config.S_churn)),
                    S_min=float(edm_cfg.get('S_min', self.gen_config.S_min)),
                    S_max=float(edm_cfg.get('S_max', self.gen_config.S_max)),
                    S_noise=float(edm_cfg.get('S_noise', self.gen_config.S_noise)),
                    lr_ups=lr_ups_M,
                    cfg_guidance=guidance_cfg if guidance_cfg.get('enabled', False) else None,
                    sigma_star=float(edm_cfg.get('sigma_star', 1.0)),
                )
            else:
                raise NotImplementedError("Currently only EDM sampler is supported in generation.")
            
            ens_model = generated.detach().cpu().float() # [M,1,H,W]

            # PMM (model space) - pmm_full is [1, 1, H, W]
            ens_for_pmm = ens_model.squeeze(1).unsqueeze(0)  # [1, M, H, W] (1 batch for pmm fn)
            pmm_full = pmm_from_ensemble(ens_for_pmm) # [1,1,H,W]

            # Back-transform to physical sapce (always compute for saving/evaluation)
            gen_phys = None
            pmm_phys = None
            hr_phys = None
            try:
                if callable(self.bt_gen):
                    gen_phys = self.bt_gen(ens_model) # [M,1,H,W] -> physical space
                    pmm_phys = self.bt_gen(pmm_full) # [1,1,H,W] -> physical space
                if callable(self.bt_hr) and (x_gen is not None):
                    hr_phys = self.bt_hr(x_gen)
            except Exception as e:
                logger.warning(f"[generation] Failed to back-transform generated samples or HR: {e}")

            # Extreme sentinel/clamp on physical ensemble (does not modify saved model-space arrays)
            if gen_phys is not None:
                try:
                    gen_phys_t = gen_phys if isinstance(gen_phys, torch.Tensor) else torch.tensor(gen_phys)
                    chk = report_precip_extremes(x_bt=gen_phys_t, name='generate_hr', cap_mm_day=self.sentinel_thr)
                    if chk.get('has_extreme', False) and self.clamp_in_gen:
                        gen_phys = torch.clamp(gen_phys_t, min=0.0, max=self.sentinel_thr)
                        logger.warning(f"[generation] Clamped extreme values >{self.sentinel_thr} mm/day in generated physical samples for date {dates[0]}")
                except Exception as e:
                    logger.warning(f"[generation] Extreme sentinel/clamp check failed: {e}")
            def _cast_phys(x):
                if x is None or not isinstance(x, torch.Tensor):
                    return x
                if self.gen_config.physical_dtype == "float16":
                    return x.half()
                return x.float()
            gen_phys = _cast_phys(gen_phys)
            pmm_phys = _cast_phys(pmm_phys)
            hr_phys = _cast_phys(hr_phys)

            # Save (model space) if requested or for backward compatibility in legacy dirs
            if save:
                if self.gen_config.save_space in ('model', 'both'):
                    _save_npz(self.out_root / 'ensembles_model' / f'{date0}.npz', ens=ens_model)  # model space 
                    _save_npz(self.out_root / 'pmm_model' / f'{date0}.npz', pmm=pmm_full)  # model space
                    logger.info("[generation] Saved ensembles_model → %s", self.out_root / 'ensembles_model' / f'{date0}.npz')
                    logger.info("[generation] Saved pmm_model → %s", self.out_root / 'pmm_model' / f'{date0}.npz')
                    # Keep legacy dirs if saving model space
                    _save_npz(self.out_root / 'ensembles' / f'{date0}.npz', ens=ens_model)  # model space 
                    _save_npz(self.out_root / 'pmm' / f'{date0}.npz', pmm=pmm_full)  # model space
                    logger.info("[generation] Saved ensembles (legacy) → %s", self.out_root / 'ensembles' / f'{date0}.npz')
                    logger.info("[generation] Saved pmm (legacy) → %s", self.out_root / 'pmm' / f'{date0}.npz')

                # Save (physical space) for evaluation
                if self.gen_config.save_space in ('physical', 'both'):
                    if gen_phys is None or pmm_phys is None:
                        logger.warning(f"[generation] Physical arrays missing; skipping saving physical space npz for date {date0}")
                    else:
                        _save_npz(self.out_root / 'ensembles_phys' / f'{date0}.npz', ens=gen_phys)  # physical space ens | pmm_model | 
                        _save_npz(self.out_root / 'pmm_phys' / f'{date0}.npz', pmm=pmm_phys)  # physical space
                        logger.info("[generation] Saved ensembles_phys → %s", self.out_root / 'ensembles_phys' / f'{date0}.npz')
                        logger.info("[generation] Saved pmm_phys → %s", self.out_root / 'pmm_phys' / f'{date0}.npz')

            def _m(x):
                return float(torch.nanmean(x)) if (x is not None and torch.is_tensor(x)) else None

            # --- after we have x_hr_1 and cond_images_gen ---

            # Save HR (model space)
            hr = x_hr_1.detach().cpu().float() if x_hr_1 is not None else None
            if save:
                _save_npz(
                    self.out_root / "lr_hr" / f"{date0}.npz",
                    hr=hr,
                    lr_local=lr_local.detach().cpu().float() if lr_local is not None else None,
                    lr_large=lr_large.detach().cpu().float() if lr_large is not None else None,
                    lr_ups_baseline=lr_ups_baseline.detach().cpu().float() if lr_ups_baseline is not None else None,
                )
            # ------------------------------------------------------------
            # Paper2+ : Save LR references using dataset-provided tensors
            # ------------------------------------------------------------
            # NOTE:
            #   - `lr_local` is the co-located 128x128 patch used for HR comparison/evaluation.
            #   - `lr_large` is the large-domain LR field (e.g. 589x789) used for context encoder/debug.
            #   - `cond_images_gen` may contain context-encoder channels and its layout is NOT a stable
            #     source of LR target channels anymore. Do not derive LR refs from `cond_images_gen`.

            # Prepare model-space references for optional in-memory return
            lr_local_model = lr_local[:1].detach().cpu().float() if (lr_local is not None) else None
            lr_large_model = lr_large[:1].detach().cpu().float() if (lr_large is not None) else None
            lr_ups_model = lr_ups_baseline[:1].detach().cpu().float() if (lr_ups_baseline is not None) else None

            # Physical-space references
            lr_local_phys = None
            lr_ups_phys = None

            if self.gen_config.save_space in ("physical", "both"):
                # LR local: inverse with LR-stats back-transform (canonical LR reference)
                if lr_local_model is not None:
                    if callable(self.bt_lr_lrspace):
                        try:
                            lr_local_phys = self.bt_lr_lrspace(lr_local_model)
                            lr_local_phys = _cast_phys(lr_local_phys)
                        except Exception as e:
                            logger.warning(f"[generation] Failed to back-transform lr_local (LR-stats) for {date0}: {e}")
                    else:
                        logger.warning(f"[generation] Missing bt_lr_lrspace; cannot invert lr_local for {date0}")

                # LR upsampled baseline: this is already in HR-zspace, so invert with bt_lr_hrspace if available
                if lr_ups_model is not None:
                    if callable(self.bt_lr_hrspace):
                        try:
                            lr_ups_phys = self.bt_lr_hrspace(lr_ups_model)
                            lr_ups_phys = _cast_phys(lr_ups_phys)
                        except Exception as e:
                            logger.warning(f"[generation] Failed to back-transform lr_ups_baseline (HR-zspace) for {date0}: {e}")
                    else:
                        # Not strictly required for evaluation; keep silent-ish
                        logger.info(f"[generation] Missing bt_lr_hrspace; skipping physical inversion of lr_ups_baseline for {date0}")

                # HR physical ref for lr_hr_phys is the canonical HR reference (single image)
                hr_phys_ref = hr
                if callable(self.bt_hr) and hr_phys_ref is not None:
                    try:
                        hr_phys_ref = self.bt_hr(hr_phys_ref)
                        hr_phys_ref = _cast_phys(hr_phys_ref)
                    except Exception as e:
                        logger.warning(f"[generation] Failed to back-transform HR reference for {date0}: {e}")
                        hr_phys_ref = None
                # Save phys refs used for evaluation
                if save:
                    _save_npz(
                        self.out_root / "lr_hr_phys" / f"{date0}.npz",
                        hr=hr_phys_ref,
                        lr_local=lr_local_phys,
                        lr_ups_baseline=lr_ups_phys,
                    )
                    logger.info("[generation] Saved lr_hr_phys → %s", self.out_root / "lr_hr_phys" / f"{date0}.npz")

                    def _m(x):
                        return float(torch.nanmean(x)) if (x is not None and torch.is_tensor(x)) else None

                    logger.info(
                        "[generation] %s LR means (phys): lr_local=%s lr_ups_baseline=%s",
                        date0, _m(lr_local_phys), _m(lr_ups_phys)
                    )

            # Canonical LR phys reference for quicklook payload
            lr_phys = lr_local_phys
            # === Optional in-memory return for quicklook ===
            if results is not None:
                # Ensure everything is CPU tensors to avoid holding GPU regs
                def _cpu(x):
                    return x.detach().cpu() if (x is not None and torch.is_tensor(x)) else x
                results.append({
                    "date": date0,
                    # model-space (compact and deterministic; physical below)
                    "ensemble_model": _cpu(ens_model),  # [M,1,H,W]
                    "pmm_model": _cpu(pmm_full),        # [1,1,H,W]
                    "hr_model": _cpu(hr),               # [1,1,H,W] or None
                    "lr_model_local": _cpu(lr_local_model),
                    "lr_model_large": _cpu(lr_large_model),
                    "lr_model_ups_baseline": _cpu(lr_ups_model),
                    # physical-space (may be large and non-deterministic due to back-transform)
                    "ensemble_phys": _cpu(gen_phys),    # [M,1,H,W]
                    "pmm_phys": _cpu(pmm_phys),         # [1,1,H,W]
                    "hr_phys": _cpu(hr_phys),           # [1,1,H,W] or None
                    "lr_phys": _cpu(lr_phys),           # [1, 1,H,W] or None
                })
                logger.info("[generation] Appended quicklook payload for %s", date0)

            n_days += 1
            # Stop early if a cap is configured
            if self.gen_config.max_dates is not None and self.gen_config.max_dates > 0 and n_days >= self.gen_config.max_dates:
                logger.info(f"[generation] Reached max_dates={self.gen_config.max_dates}, stopping generation early.")
                break

        # Manifest for reproducibility
        manifest = {
            'model': model_name,
            'ensemble_size': M,
            'sampler_kind': self._sampler_kind,
            'sampler_steps': steps,
            'seed': int(self.cfg.get('evaluation', {}).get('seed', self.gen_config.seed)),
            'n_days': n_days,
            'save_space': self.gen_config.save_space,
            'physical_dtype': self.gen_config.physical_dtype,
            'stationary_cutout_cfg': bool(self.stationary_cutout),
            'lsm_stationary_observed': bool(self._lsm_stationary_ok and (self._first_lsm is not None)),
        }
        if save:
            (self.out_root / 'meta').mkdir(parents=True, exist_ok=True)
            (self.out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            logger.info("[generation] Manifest path → %s", self.out_root / 'meta' / 'manifest.json')
        else:
            logger.info("[generation] save=False (quicklook) → skipping manifest write")
        logger.info(f"[generation] Done. Wrote {n_days} days to {self.out_root}. Manifest: {manifest}")


        if output_results:
            logger.info("[generation] Returning in-memory results for %d day(s)", len(results) if results is not None else 0)
            return {
                "manifest": manifest,
                "out_root": str(self.out_root),
                "results": results
            }
