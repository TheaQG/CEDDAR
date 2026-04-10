"""
    TODO:
        - Implement mixed precision training 
        - Make precipitation evaluations only when precipitation is the target variable
"""

import os
import torch
import copy
import pickle
import tqdm
import logging 
import math
import time 

import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import Optional
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

from sbgm.heads.rain_gate import RainGate
from sbgm.special_transforms import build_back_transforms_from_stats, lr_baseline_to_hr_zspace
from sbgm.utils import get_model_string, extract_samples, crop_bounds_to_stats_str
from sbgm.plotting_utils import (
    get_cmaps,
    plot_samples_and_generated,
    plot_live_training_metrics,
    plot_fss_history,
    plot_psd_slope_history,
    plot_quantiles_wetday_history,
    plot_training_monitor_generated
    )
from sbgm.monitoring import (
    report_precip_extremes,
    compute_fss_at_scales,
    compute_psd_slope,
    compute_p95_p99_and_wet_day,
    tensor_stats,
    save_histogram,
    plot_saved_histograms,
    in_loop_metrics,
    _save_weight_map_viz,
    _plot_reliability_curve
    )
from sbgm.score_sampling import Euler_Maruyama_sampler, pc_sampler, ode_sampler, edm_sampler
from sbgm.training_utils import get_loss_fn, apply_cfg_dropout
from sbgm.variable_utils import get_units

# Speed up conv algo selection on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
'''
    ToDo:
        - Add support for mixed precision training
        - Add support for EMA (Exponential Moving Average) of the model
        - Add support for custom weight initialization
'''

# Set up logging
logger = logging.getLogger(__name__)


class TrainingPipeline_general:
    '''
        Class for building a training pipeline for the SBGM.
        To run through the training batches in one epoch.
    '''
    def _model_ref(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _dist_barrier(self) -> None:
        if self.distributed and dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _timer_enabled(self) -> bool:
        mon = self.cfg.get('monitoring', {}) if isinstance(self.cfg, dict) else {}
        perf = mon.get('performance_timing', {}) if isinstance(mon, dict) else {}
        return bool(perf.get('enabled', False))
    
    def _init_perf_state(self) -> None:
        self.perf_state = {
            'cond_img_calls': 0,
            'cond_img_total_sec': 0.0,
            'train_data_wait_calls': 0,
            'train_data_wait_total_sec': 0.0,
            'train_forward_calls': 0,
            'train_forward_total_sec': 0.0,
            'train_backward_calls': 0,
            'train_backward_total_sec': 0.0,
            'train_step_calls': 0,
            'train_step_total_sec': 0.0,
            'train_batch_calls': 0,
            'train_batch_total_sec': 0.0,
        }

    def _reset_perf_state(self) -> None:
        self._init_perf_state()
    
    def _record_perf_time(self, key: str, dt: float) -> None:
        if not getattr(self, 'perf_state', None):
            return
        total_key = f"{key}_total_sec"
        calls_key = f"{key}_calls"
        if total_key not in self.perf_state:
            self.perf_state[total_key] = 0.0
        if calls_key not in self.perf_state:
            self.perf_state[calls_key] = 0
        self.perf_state[total_key] += float(dt)
        self.perf_state[calls_key] += 1
    
    def log_perf_summary(self, prefix: str = "[perf]") -> None:
        if not getattr(self, 'perf_state', None):
            return

        def _vals(name: str):
            calls = int(self.perf_state.get(f'{name}_calls', 0))
            total = float(self.perf_state.get(f'{name}_total_sec', 0.0))
            mean = total / max(calls, 1)
            return calls, total, mean

        c_calls, c_total, c_mean = _vals('cond_img')
        dw_calls, dw_total, dw_mean = _vals('train_data_wait')
        f_calls, f_total, f_mean = _vals('train_forward')
        b_calls, b_total, b_mean = _vals('train_backward')
        s_calls, s_total, s_mean = _vals('train_step')
        bt_calls, bt_total, bt_mean = _vals('train_batch')

        logger.info(
            "%s cond_img_build: calls=%s total_sec=%.4f mean_sec=%.4f | "
            "data_wait: calls=%s total_sec=%.4f mean_sec=%.4f | "
            "forward: calls=%s total_sec=%.4f mean_sec=%.4f | "
            "backward: calls=%s total_sec=%.4f mean_sec=%.4f | "
            "step: calls=%s total_sec=%.4f mean_sec=%.4f | "
            "batch_total: calls=%s total_sec=%.4f mean_sec=%.4f",
            prefix,
            c_calls, c_total, c_mean,
            dw_calls, dw_total, dw_mean,
            f_calls, f_total, f_mean,
            b_calls, b_total, b_mean,
            s_calls, s_total, s_mean,
            bt_calls, bt_total, bt_mean,
        )

    def _build_cond_img(self, batch: dict) -> torch.Tensor | None:
        """
        Build the conditioning image passed to the UNet encoder.

        Supports Paper2 spatial context encoder:
          - large-domain LR fields: {var}_lr  -> ContextEncoder -> ctx [B,Cctx,128,128]
          - local co-located LR fields: {var}_lr_local (fallback to {var}_lr if missing)

        Config:
          paper2.spatial_context.encoder.enabled: bool
          paper2.spatial_context.encoder.input_mode: 'context_only' | 'context_plus_local'
        """
        t0 = time.perf_counter() if self._timer_enabled() else None
        try:
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
                            f"paper2.spatial_context.mode='large_domain' requires '{k_local}' in batch "
                            f"(got keys: {list(batch.keys())})."
                        )
                    lr_tensors_local.append(batch[k_local])
                else:
                    if (k_local in batch) and (batch[k_local] is not None):
                        lr_tensors_local.append(batch[k_local])
                    else:
                        lr_tensors_local.append(batch[k_ctx])

            cond_local = torch.cat(lr_tensors_local, dim=1).to(self.device)

            paper2 = (self.cfg.get("paper2", {}) or {})
            spatial = (paper2.get("spatial_context", {}) or {})
            enc_cfg = (spatial.get("encoder", {}) or {})
            model_ref = self._model_ref()
            use_ctx = bool(enc_cfg.get("enabled", False)) and (getattr(model_ref, "context_encoder", None) is not None)
            ctx_mode = str(enc_cfg.get("input_mode", "context_plus_local"))

            if not use_ctx:
                return cond_local

            xs = []
            for v in self.lr_vars:
                k = f"{v}_lr"
                t = batch[k]
                xs.append(t[:, 0])
            x_bvhw = torch.stack(xs, dim=1)
            x_ctx = x_bvhw.unsqueeze(1).to(self.device)

            ctx = self._model_ref().encode_spatial_context(x_ctx)  # [B,Cctx,128,128]

            if ctx_mode == "context_only":
                return ctx

            return torch.cat([cond_local, ctx], dim=1)
        finally:
            if t0 is not None:
                self._record_perf_time('cond_img', time.perf_counter() - t0)

    def _build_local_cond_img(self, batch: dict) -> torch.Tensor | None:
        """
        Build ONLY the local/co-located LR conditioning tensor at HR resolution.

        This intentionally excludes Paper2 context-encoder features and is used for:
          - RainGate inputs
          - LR baseline extraction for EDM residual prediction
          - diagnostics that should operate on local LR channels only
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
                        f"paper2.spatial_context.mode='large_domain' requires '{k_local}' in batch "
                        f"(got keys: {list(batch.keys())})."
                    )
                t = batch[k_local]
            else:
                if (k_local in batch) and (batch[k_local] is not None):
                    t = batch[k_local]
                else:
                    t = batch[k_ctx]

            if not torch.is_tensor(t):
                t = torch.tensor(t)
            t = t.to(self.device)
            if t.ndim == 3:
                t = t.unsqueeze(1)
            elif t.ndim == 2:
                t = t.unsqueeze(0).unsqueeze(0)
            lr_tensors_local.append(t)

        return torch.cat(lr_tensors_local, dim=1)

    @staticmethod
    def _build_cond_channel_map_cfg(lr_vars: list, hr_var: str, dual_lr: bool, lr_main_var_scale: str) -> dict:
        """
        Mirror the dataset's LR cond channel layout in training-side logic.

        Conventions:
          - If dual_lr and target variable exists in lr_vars: that variable expands to 2 channels:
                * "main"    : scaled with lr_main_var_scale (HR/LR/HR_LR)
                * "lr_only" : scaled with LR stats
          - All other LR variables are 1 channel ("main")
        """
        def _space_tag(scale: str) -> str:
            s = str(scale).upper()
            if s == "LR":
                return "lr_stats"
            if s == "HR":
                return "hr_stats"
            if s in ("HR_LR", "LR_HR", "COMBINED", "BOTH"):
                return "combo_stats"
            return "lr_stats"

        order = list(lr_vars)
        slices = {}
        spaces = {}

        # Identify main LR condition (prefer exact match; fallback aliases)
        main_lr_cond = None
        if hr_var in order:
            main_lr_cond = hr_var
        else:
            alias_pairs = [('prcp', 'tp'), ('tp', 'prcp'), ('temp', 't2m'), ('t2m', 'temp')]
            for hr_name, lr_alias in alias_pairs:
                if hr_var == hr_name and lr_alias in order:
                    main_lr_cond = lr_alias
                    break

        c = 0
        for cond in order:
            if bool(dual_lr) and (main_lr_cond is not None) and (cond == main_lr_cond):
                slices[cond] = {"main": (c, c + 1), "lr_only": (c + 1, c + 2)}
                spaces[cond] = {"main": _space_tag(lr_main_var_scale), "lr_only": "lr_stats"}
                c += 2
            else:
                slices[cond] = {"main": (c, c + 1)}
                spaces[cond] = {"main": "lr_stats"}
                c += 1

        return {"order": order, "main_lr_cond": main_lr_cond, "slices": slices, "spaces": spaces, "n_channels_total": c}

    def _cond_slice(self, var: str, kind: str = "main") -> tuple[int, int]:
        """
        Return (start, end) channel slice indices for a given LR variable and kind.
        kind: "main" or "lr_only" (only exists for the main LR var when dual_lr=True).
        """
        cmap = getattr(self, "cond_channel_map_cfg", None)
        if cmap is None:
            raise RuntimeError("cond_channel_map_cfg is not initialized.")
        if var not in cmap["slices"]:
            raise KeyError(f"Variable '{var}' not found in cond_channel_map_cfg slices. Available: {list(cmap['slices'].keys())}")
        if kind not in cmap["slices"][var]:
            raise KeyError(f"Kind '{kind}' not available for var '{var}'. Available: {list(cmap['slices'][var].keys())}")
        return cmap["slices"][var][kind]

    def __init__(self,
                 model,
                 marginal_prob_std_fn,
                 diffusion_coeff_fn,
                 optimizer,
                 device,
                 lr_scheduler,
                 cfg
                 ):
        '''
            Initialize the training pipeline.
            Args:
                model: PyTorch model to be trained. 
                loss_fn: Loss function for the model. 
                optimizer: Optimizer for the model.
                device: Device to run the model on.
                weight_init: Weight initialization method.
                custom_weight_initializer: Custom weight initialization method.
                sdf_weighted_loss: Boolean to use SDF weighted loss.
                with_ema: Boolean to use Exponential Moving Average (EMA) for the model.
        '''
        # Store the full configuration for later use
        self.cfg = cfg
        runtime = cfg.get('runtime', {}) if isinstance(cfg, dict) else {}
        self.distributed = bool(runtime.get('distributed', False))
        self.rank = int(runtime.get('rank', 0))
        self.local_rank = int(runtime.get('local_rank', 0))
        self.world_size = int(runtime.get('world_size', 1))
        self.is_main_process = bool(runtime.get('is_main_process', self.rank == 0))
        self._init_perf_state()

        self.writer = None  # Placeholder for TensorBoard writer, if needed

        # Set class variables
        self.model = model
        # Set debug_pre_sigma_div on the raw model (important under DDP)
        self._model_ref().debug_pre_sigma_div = cfg['training'].get('debug_pre_sigma_div', False)

        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.diffusion_coeff_fn = diffusion_coeff_fn
        self.optimizer = optimizer
        # self.loss_fn = loss_fn
        self.loss_fn = get_loss_fn(self.cfg, marginal_prob_std_fn_in=getattr(self, 'marginal_prob_std_fn', None))

        self.lr_scheduler = lr_scheduler

        self.scaling = cfg['transforms']['scaling']
        self.global_prcp_eps = cfg['transforms'].get('prcp_eps', 0.01)

        self.hr_var = cfg['highres']['variable']
        self.hr_scaling_method = cfg['highres']['scaling_method']
        self.full_domain_dims_hr = cfg['highres']['full_domain_dims']
        self.crop_region_hr = cfg['highres']['cutout_domains']

        self.lr_vars = cfg['lowres']['condition_variables']
        self.lr_scaling_methods = cfg['lowres']['scaling_methods']

        # Training-side mirror of the dataset LR-channel layout (critical for dual_lr and multi-var conditioning)
        self.cond_channel_map_cfg = self._build_cond_channel_map_cfg(
            lr_vars=self.lr_vars,
            hr_var=self.hr_var,
            dual_lr=bool(cfg['lowres'].get('dual_lr', False)),
            lr_main_var_scale=str(cfg['lowres'].get('lr_main_var_scale', 'LR')),
        )
        if self.is_main_process:
            logger.info(
                "[train] LR channel layout (cfg mirror):\n"
                f"  lr_vars            : {self.cond_channel_map_cfg['order']}\n"
                f"  main_lr_cond       : {self.cond_channel_map_cfg['main_lr_cond']}\n"
                f"  slices             : {self.cond_channel_map_cfg['slices']}\n"
                f"  spaces             : {self.cond_channel_map_cfg['spaces']}\n"
                f"  n_channels_total   : {self.cond_channel_map_cfg['n_channels_total']}"
            )
        self.full_domain_dims_lr = cfg['lowres']['full_domain_dims']
        self.crop_region_lr = cfg['lowres']['cutout_domains']

        # Cache strings for stats lookups
        self._dom_hr_str = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        self._dom_lr_str = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        self._crop_hr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx")

        paper2 = (self.cfg.get('paper2', {}) or {})
        spatial = (paper2.get('spatial_context', {}) or {})
        spatial_mode = str(spatial.get('mode', '')).lower()
        if spatial_mode == 'large_domain' and (self.full_domain_dims_lr is not None):
            self._crop_lr_str = crop_bounds_to_stats_str(
                [0, self.full_domain_dims_lr[1], 0, self.full_domain_dims_lr[0]],
                order="xxyy",
            )
        elif spatial_mode == 'colocated':
            self._crop_lr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx")
        else:
            self._crop_lr_str = crop_bounds_to_stats_str(self.crop_region_lr, order="xxyy")

        self._stats_root = self.cfg['paths']['stats_load_dir']
        self._hr_method_for_target = self.hr_scaling_method
        # Assume LR scaling methods is a list aligned with lr_vars; get method for the target variable
        if self.hr_var in self.lr_vars:
            idx_t = self.lr_vars.index(self.hr_var)
            self._lr_method_for_target = self.lr_scaling_methods[idx_t]
            if self.is_main_process:
                logger.info(f"Determined LR scaling method for target variable '{self.hr_var}': {self._lr_method_for_target}")
        else:
            self._lr_method_for_target = None
            if self.is_main_process:
                logger.warning(f"HR target variable '{self.hr_var}' not found in LR condition variables {self.lr_vars}. Cannot determine LR scaling method for target - residuals may not be aligned.")
        # inject into dicts
        self.bt_gen_key = "generated"

        # --------------------------------------------------- assemble key order
        self.bt_hr_key = f"{self.hr_var}_hr"
        self.bt_lr_keys = [f"{var}_lr" for var in self.lr_vars]

        self.weight_init = cfg['training']['weight_init']
        self.custom_weight_initializer = cfg['training']['custom_weight_initializer']
        self.sdf_weighted_loss = cfg['training']['sdf_weighted_loss']

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # EMA parameters
        self.with_ema = cfg['training']['with_ema']
        self.ema_decay = float(cfg['training'].get('ema_decay', 0.9999)) # Default to 0.9999 if not specified
        self.ema_warmup_steps = int(cfg.get("training", {}).get("ema_warmup_steps", 0))
        self.global_step = 0
        # EMA debug logging (theta vs theta_ema)
        self.ema_debug_every = int(cfg.get("training", {}).get("ema_debug_every", 200))
        self.ema_debug_n_params = int(cfg.get("training", {}).get("ema_debug_n_params", 5))
        if self.with_ema:
            self._init_ema()

        # RainGate configuration (auxiliary wet/dry head and optional pixel reweighting)
        rg_cfg = self.cfg.get('rain_gate', {})
        self.rain_gate_enabled = bool(rg_cfg.get('enabled', False))
        self.rain_gate_reweight_enabled = bool(rg_cfg.get('reweight_enabled', False))

        # Defaults chosen to be safe if section is missing
        self.rain_gate_loss_weight = float(rg_cfg.get('loss_weight_bce', 0.0))
        self.rain_gate_threshold_mm = float(rg_cfg.get('wet_threshold_mm', 1.0))

        # Pixel weighting shape/strength hyperparameters
        self.rg_alpha = float(rg_cfg.get('alpha', 1.0))   # strength of upweighting wet pixels
        self.rg_gamma = float(rg_cfg.get('gamma', 1.0))   # curvature of weighting function

        # Instantiate RainGate network and its loss, if enabled
        self.rain_gate = None
        self.rain_gate_criterion = None
        if self.rain_gate_enabled:
            if self.is_main_process:
                logger.info("→ RainGate auxiliary head enabled")
            # Input channels for RainGate: start with LR channels; geo channels can be added later in the training step
            # We don't know the exact channel count here yet, so we create the module lazily on first use.
            # For now we only store the configuration; the object will be built in the training loop when shapes are known.
            self.rain_gate_lazy_init = True
        else:
            self.rain_gate_lazy_init = False

        # Classifier free guidance config
        self.cfg_guidance = self.cfg.get('classifier_free_guidance', {})
        if self.cfg_guidance.get('enabled', False) and self.is_main_process:
            logger.info("→ Classifier-free guidance enabled")
            logger.info(f"      → drop_prob_lr: {self.cfg_guidance.get('drop_prob_lr', 0.1)}")
            logger.info(f"      → drop_prob_geo   = {self.cfg_guidance.get('drop_prob_geo', self.cfg_guidance.get('drop_prob_lr', 0.1))}")
            logger.info(f"      → drop_prob_class = {self.cfg_guidance.get('drop_prob_class', 0.0)}")
            logger.info(f"      → null_lr_strategy= {self.cfg_guidance.get('null_lr_strategy','zero')} (scalar={self.cfg_guidance.get('null_lr_scalar',0.0)})")
            logger.info(f"      → null_geo_value  = {self.cfg_guidance.get('null_geo_value', -5.0)}")
            logger.info(f"      → null_label_id   = {self.cfg_guidance.get('null_label_id', 0)}, null_scalar_value = {self.cfg_guidance.get('null_scalar_value', 0.0)}")


        # Initialize weights if needed
        if self.weight_init:
            if self.custom_weight_initializer is not None:
                # Use custom weight initializer if provided
                self._model_ref().apply(self.custom_weight_initializer)
            else:
                self._model_ref().apply(self.xavier_init_weights)
            if self.is_main_process:
                logger.info(f"→ Model weights initialized with {self.custom_weight_initializer.__name__ if self.custom_weight_initializer else 'Xavier uniform'} initialization.")

        # Set up checkpoint directory, name and path
        self.checkpoint_dir = cfg['paths']['checkpoint_dir']
        self.checkpoint_name = get_model_string(cfg) + '.pth.tar' 
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)

        # Create the checkpoint directory if it does not exist
        checkpoint_dir_existed = os.path.exists(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.is_main_process:
            if checkpoint_dir_existed:
                logger.info(f"→ Checkpoint directory already exists at {self.checkpoint_dir}")
            else:
                logger.info(f"→ Checkpoint directory created at {self.checkpoint_dir}")

        # Set the model string based on the configuration
        self.model_string = get_model_string(cfg)

        # Set path to figures, samples, losses
        self.path_samples = cfg['paths']['path_save'] + '/samples/' + self.model_string
        self.path_losses = cfg['paths']['path_save'] + '/losses'
        self.path_figures = self.path_samples + '/Figures'
        # Metrics path
        self.path_metrics = os.path.join(self.path_figures, 'metrics')

        # Create the directories if they do not exist
        samples_existed = os.path.exists(self.path_samples)
        losses_existed = os.path.exists(self.path_losses)
        figures_existed = os.path.exists(self.path_figures)
        metrics_existed = os.path.exists(self.path_metrics)

        os.makedirs(self.path_samples, exist_ok=True)
        os.makedirs(self.path_losses, exist_ok=True)
        os.makedirs(self.path_figures, exist_ok=True)
        os.makedirs(self.path_metrics, exist_ok=True)
        
        # Debug/diagnostics figures directory
        self.path_diagnostics = os.path.join(self.path_metrics, 'debug')
        diagnostics_existed = os.path.exists(self.path_diagnostics)
        os.makedirs(self.path_diagnostics, exist_ok=True)

        if self.is_main_process:
            if not samples_existed:
                logger.info(f"→ Samples directory created at {self.path_samples}")
            if not losses_existed:
                logger.info(f"→ Losses directory created at {self.path_losses}")
            if not figures_existed:
                logger.info(f"→ Figures directory created at {self.path_figures}")
            if not metrics_existed:
                logger.info(f"→ Metrics directory created at {self.path_metrics}")
            if not diagnostics_existed:
                logger.info(f"→ Diagnostics directory created at {self.path_diagnostics}")



        # === Monitoring: extreme precipitation values in generated samples ===
        monitor_cfg = cfg.get('monitoring', {})
        monitor_prcp = monitor_cfg.get('extreme_prcp', {})
        self.extreme_enabled = bool(monitor_prcp.get('enabled', True))
        self.extreme_threshold_mm = float(monitor_prcp.get('threshold_mm', 500.0)) # Threshold in mm for extreme precipitation
        self.extreme_every_step = int(monitor_prcp.get('every_steps', 50)) # Monitor every n steps
        self.extreme_backtransform = bool(monitor_prcp.get('back_transform', True)) # Backtransform samples before checking extremes
        self.extreme_log_first_n = int(monitor_prcp.get('log_first_n', 5)) # Log the first n extreme values in detail
        self.extreme_in_validation = bool(monitor_prcp.get('check_in_validation', True)) # Check extreme values in validation set as well
        self.extreme_clamp_in_gen = bool(monitor_prcp.get('clamp_in_generation', True)) # Clamp extreme values in generated samples to threshold

        try:
            full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
            full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
            crop_region_hr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx")
            # Sentinel / monitoring back-transforms must use the effective LR stats domain,
            # not the raw lowres.cutout_domains from config.
            # - large_domain: use full LR domain [x1, x2, y1, y2] = [0, W, 0, H]
            # - colocated: use HR crop string convention because LR stats were saved that way
            paper2 = (cfg.get('paper2', {}) or {})
            spatial = (paper2.get('spatial_context', {}) or {})
            spatial_mode = str(spatial.get('mode', '')).lower()
            if spatial_mode == 'large_domain' and (self.full_domain_dims_lr is not None):
                lr_crop_bounds_eff = [0, self.full_domain_dims_lr[1], 0, self.full_domain_dims_lr[0]]
                crop_region_lr_str = crop_bounds_to_stats_str(lr_crop_bounds_eff, order="xxyy")
            elif spatial_mode == 'colocated':
                lr_crop_bounds_eff = self.crop_region_hr
                crop_region_lr_str = crop_bounds_to_stats_str(lr_crop_bounds_eff, order="yyxx")
            else:
                lr_crop_bounds_eff = self.crop_region_lr
                crop_region_lr_str = crop_bounds_to_stats_str(lr_crop_bounds_eff, order="xxyy")

            self.back_transforms_train = build_back_transforms_from_stats(
                hr_var=self.hr_var,
                hr_model=cfg['highres']['model'],
                domain_str_hr=full_domain_dims_str_hr,
                crop_region_str_hr=crop_region_hr_str,
                hr_scaling_method=self.hr_scaling_method,
                hr_buffer_frac=cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                lr_vars=self.lr_vars,
                lr_model=cfg['lowres']['model'],
                lr_scaling_methods=self.lr_scaling_methods,
                domain_str_lr=full_domain_dims_str_lr,
                crop_region_str_lr=crop_region_lr_str,
                lr_buffer_frac=cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                split=cfg['transforms'].get('scaling_split', 'train'),
                stats_dir_root=cfg['paths']['stats_load_dir'],
                eps=self.global_prcp_eps
            )
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"[monitor] Could not build back transforms for sentinel; will skip back_transform in training. Error: {e}")
            self.back_transforms_train = None


        # === EDM flags (used for residual baseline handling) ===
        self.edm_enabled = bool((cfg.get('edm', {}).get('enabled', False)))
        self.edm_predict_residual = bool((cfg.get('edm', {}).get('predict_residual', False)))

        # === Live, lightweight monitors (append in loop; plot occasionally) ===
        moncfg = cfg.get('monitoring', {})
        self.monitor_plot_every_n_epochs = int(moncfg.get('plot_every_n_epochs', 5))
        self.live_metrics = {
            'steps': [],
            'edm_cosine': [],
            'hr_lr_corr': []
        }
        self.eval_land_only = bool(cfg.get('evaluation', {}).get('eval_land_only', False))

        # Persistent histories of epoch-level monitors
        self.fss_hist: list[dict] = []
        self.psd_hist: list[dict] = []
        self.q_hist: list[dict] = []
        self.epoch_list = []

        # Monitoring configuration
        cfg_mon__end_of_epoch = moncfg.get('end_of_epoch', {})
        self.fss_scales_km = list(cfg_mon__end_of_epoch.get('fss_km', [5, 10, 20])) # Scales in km for FSS
        self.fss_threshold_mm = float(cfg_mon__end_of_epoch.get('fss_threshold_mm', 1.0)) # Threshold in mm for FSS
        self.pixel_km = float(cfg_mon__end_of_epoch.get('grid_km_per_px', 2.5)) # Grid spacing in km/px
        self.wetday_thresh = float(cfg_mon__end_of_epoch.get('wet_day_threshold_mm', 0.1)) # Wet day threshold in mm/day
        self.psd_compare_to_hr = bool(cfg_mon__end_of_epoch.get('psd_compare_to_hr', True)) # Whether to compare LR/HR PSD slopes
        self.quantiles_compare_to_hr = bool(cfg_mon__end_of_epoch.get('quantiles_compare_to_hr', True)) # Whether to compare LR/HR quantiles

        # === Rain/not-rain gating mini-head (optional) ===
        rg_cfg = cfg.get('rain_gate', {})
        self.rg_enabled = bool(rg_cfg.get('enabled', False))
        self.rg_include_lsm = bool(rg_cfg.get('include_lsm', True))
        self.rg_include_topo = bool(rg_cfg.get('include_topo', True))
        self.rg_include_lr_baseline = bool(rg_cfg.get('include_lr_baseline', True)) # optional
        self.rg_threshold_mm = float(rg_cfg.get('wet_threshold_mm', cfg.get('monitoring', {}).get('end_of_epoch', {}).get('wet_day_threshold_mm', 0.1))) # Default to wet day threshold if not specified
        self.rg_threshold_modelSpace = float(rg_cfg.get('wet_threshold_modelSpace', 0.1)) # Threshold in model space (e.g. z-score) for wet/dry classification when computing BCE loss


        # Reweighting of loss based on rain gate prediction
        self.rg_reweight_enabled = bool(rg_cfg.get('reweight_enabled', False)) # Whether to reweight loss based on rain gate prediction
        self.rg_warm_start = int(rg_cfg.get('reweight_warm_start_epochs', 5)) # Number of epochs to wait before starting reweighting
        self.rg_ramp = int(rg_cfg.get('reweight_ramp_epochs', 0)) # Number of epochs over which to ramp up reweighting from 0 to full
        self.rg_loss_weight = float(rg_cfg.get('loss_weight_bce', 0.1)) # Weight of the BCE loss for rain gate
        self.rg_pos_weight = float(rg_cfg.get('pos_weight', 2.0)) # Positive class weight for BCE loss to handle class imbalance
        self.rg_lr = float(rg_cfg.get('learning_rate', self.optimizer.param_groups[0]['lr'] if self.optimizer is not None else 1e-4)) # Learning rate for rain gate head

        self.rain_gate: RainGate | None = None
        if self.rg_enabled:
            # Determine input channel count from config
            c_in = 0
            # LR condition channels (already upsampled to HR in dataset) — respect dual-LR expansion
            lr_c = int(self.cond_channel_map_cfg["n_channels_total"])
            c_in += lr_c
            # Optional static inputs
            if self.rg_include_lsm:
                c_in += 1
            if self.rg_include_topo:
                c_in += 1  # NOTE: Later add slope
            # Only include LR baseline channel when EDM residual prediction is active
            if self.rg_include_lr_baseline and self.edm_enabled and self.edm_predict_residual:
                c_in += 1
            self.rain_gate = RainGate(c_in=c_in, c_hidden=int(rg_cfg.get('c_hidden', 16)))
            self.rain_gate.to(self.device)
            # Attach rain_gate params to existing optimizer as a new param group
            if self.optimizer is not None:
                self.optimizer.add_param_group({'params': self.rain_gate.parameters(), 'lr': self.rg_lr})
                import torch.optim.lr_scheduler as _schedulers
                if isinstance(self.lr_scheduler, _schedulers.ReduceLROnPlateau):
                    n_groups = len(self.optimizer.param_groups)
                    # Extend min_lrs to match new param group count
                    if len(self.lr_scheduler.min_lrs) < n_groups:
                        tail = self.lr_scheduler.min_lrs[-1] if len(self.lr_scheduler.min_lrs) > 0 else 0.0
                        self.lr_scheduler.min_lrs += [tail] * (n_groups - len(self.lr_scheduler.min_lrs))
        else:
            self.rain_gate = None
            c_in = 0
        
        if self.is_main_process:
            logger.info(f"→ Rain gating head enabled: {self.rg_enabled}, c_in: {c_in if self.rg_enabled else 'N/A'}")

    def _check_y_runtime(self, y: torch.Tensor | None) -> None:
        """Runtime safeguard for seasonal label just before model forward (train and eval)"""
        if y is None:
            return
        use_sincos = bool(self.cfg.get('stationary_conditions', {}).get('seasonal_conditions', {}).get('use_sin_cos_embedding', False))
        if use_sincos:
            assert torch.is_floating_point(y), f"[DOY-check/train] expected float y for sin/cos; got: {y.dtype}"
            assert y.ndim == 2 and y.shape[1] == 2, f"[DOY-check/train] expected shape [B, 2] for sin/cos; got: {tuple(y.shape)}"
            m = float(torch.min(y)); M = float(torch.max(y))
            assert (m >= -1.05) and (M <= 1.05), f"[DOY-check/train] expected sin/cos in [-1, 1]; got min {m}, max {M}"
        else:
            assert y.dtype in (torch.long, torch.int64), f"[DOY-check/train] expected int64/long y for class labels; got: {y.dtype}"
            assert (y.ndim == 1) or (y.ndim == 2 and y.shape[1] == 1), f"[DOY-check/train] expected shape [B] or [B, 1] for class labels; got: {tuple(y.shape)}"

    def _build_lr_ups_baseline(self, cond_images: torch.Tensor | None):
        """
            Extract LR baseline channel (same variable as HR target) from cond_images and upsample to HR resolution.
            Ensure it is expressed in HR z-space (or HR min-max space) before using for residual EDM.
            Returns [B, 1, H, W] or raises if unavailable when predict_residual is True.
        """
        if cond_images is None:
            raise ValueError("cond_images is None, cannot extract LR baseline for residual prediction.")
        
        target_var = self.hr_var
        
        # Determine which LR var name matches the HR target (supports aliases like prcp↔tp, temp↔t2m)
        cmap = self.cond_channel_map_cfg
        main_lr_cond = cmap.get("main_lr_cond", None)
        if main_lr_cond is None:
            raise ValueError(
                f"Could not identify a main LR condition matching HR target '{target_var}' within {self.lr_vars}. "
                "Cannot extract LR baseline for residual prediction."
            )

        # Choose which baseline channel to use:
        #   - If baseline_space == 'lr' → prefer LR-stats channel (lr_only when dual_lr)
        #   - Else (hr/auto) → if a HR-stats-scaled main channel exists, prefer it (no conversion needed),
        #                     otherwise use LR-stats channel and convert to HR space.
        baseline_space = str(self.cfg.get('edm', {}).get('baseline_space', 'hr')).lower()
        dual_lr = bool(self.cfg.get('lowres', {}).get('dual_lr', False))
        lr_main_var_scale = str(self.cfg.get('lowres', {}).get('lr_main_var_scale', 'LR')).upper()

        use_kind = "main"
        if dual_lr and (main_lr_cond in cmap["slices"]) and ("lr_only" in cmap["slices"][main_lr_cond]):
            if baseline_space == "lr":
                use_kind = "lr_only"
            else:
                # Prefer HR-scaled main channel when requested and configured
                if lr_main_var_scale == "HR" and cmap["spaces"][main_lr_cond].get("main", "") == "hr_stats":
                    use_kind = "main"
                else:
                    use_kind = "lr_only"

        s0, s1 = self._cond_slice(main_lr_cond, kind=use_kind)
        if cond_images.shape[1] < s1:
            raise ValueError(
                f"cond_images has shape {tuple(cond_images.shape)} but needs at least {s1} channels to extract "
                f"baseline slice for '{main_lr_cond}:{use_kind}' = ({s0},{s1})."
            )

        lr_chan = cond_images[:, s0:s1, :, :]  # [B, 1, H, W] (already upsampled to HR size)
        logger.debug(f"[baseline] extracted LR baseline from cond_images var='{main_lr_cond}', kind='{use_kind}', baseline_space='{baseline_space}'")

        # If baseline_space is explicitly 'lr', return as-is (caller will treat it as LR-space baseline)
        if baseline_space == 'lr':
            logger.debug("[baseline] baseline_space='lr' → returning baseline channel as-is (LR-space / already-upsampled).")
            return lr_chan

        # If the extracted baseline channel is already HR-stats normalized, do NOT convert again.
        # This happens for dual-LR when lr_main_var_scale='HR' and we picked kind='main'.
        space_tag = self.cond_channel_map_cfg["spaces"][main_lr_cond].get(use_kind, "lr_stats")
        if space_tag == "hr_stats":
            logger.debug("[baseline] baseline channel is already in HR-stats space → returning as-is (no LR→HR conversion).")
            return lr_chan

        # Else, convert LR-stats-normalized baseline to HR-stats space via lr_baseline_to_hr_zspace
        lr_method_for_baseline = self._lr_method_for_target 
        if lr_method_for_baseline is None:
            raise ValueError("LR scaling method for baseline is None. Cannot proceed with lr_baseline_to_hr_zspace. Please check your configuration.")

        lr_in_hr_space = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_chan,
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
            eps=self.global_prcp_eps
        )

        return lr_in_hr_space

    def _assert_all_finite(self, name, t):
        if t is not None and not torch.isfinite(t).all():
            mn = t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else float('nan')
            mx = t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else float('nan')
            raise ValueError(f"Input '{name}' contains non-finite values. Min: {mn}, Max: {mx}")


    def xavier_init_weights(self, m):
        '''
            Xavier weight initialization.
            Args:
                m: Model to initialize weights for.
        '''

        # Check if the layer is a linear or convolutional layer
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize weights with Xavier uniform
            nn.init.xavier_uniform_(m.weight)
            # If model has bias, initialize with 0.01 constant
            if m.bias is not None and torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)
    
    def _init_ema(self):
        """ 
            Initialize Exponential Moving Average (EMA) model as a deepcopy of the raw model and freeze it.
        """
        raw_model = self._model_ref()
        self.ema_model = copy.deepcopy(raw_model)
        self.ema_model.to(self.device)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        if self.is_main_process:
            logger.info(f"→ EMA model initialized with decay {self.ema_decay}")

    @torch.no_grad()
    def _update_ema(self):
        """
            Exponential moving average (EMA) update: ema = d*ema + (1-d)*model
        """
        if not getattr(self, 'ema_model', None):
            return
        d = self.ema_decay
        msd = self._model_ref().state_dict()
        esd = self.ema_model.state_dict()
        for k in esd.keys():
            if k in msd and esd[k].dtype.is_floating_point:
                esd[k].mul_(d).add_(msd[k].detach(), alpha=1 - d)
            elif k in msd:
                esd[k].copy_(msd[k].detach())
    
    def _log_ema_delta(self):
        """Log ||theta - theta_ema|| / ||theta|| for a few parameters (sanity check)."""
        if (not getattr(self, "with_ema", False)) or (not hasattr(self, "ema_model")) or (not self.is_main_process):
            return

        try:
            msd = self._model_ref().state_dict()
            esd = self.ema_model.state_dict()

            rels = []
            count = 0

            for k, v in msd.items():
                if not torch.is_tensor(v):
                    continue
                if not torch.is_floating_point(v):
                    continue
                if k not in esd:
                    continue
                ve = esd[k]
                if (not torch.is_tensor(ve)) or (not torch.is_floating_point(ve)):
                    continue

                dv = (v.detach() - ve.detach())
                num = torch.linalg.norm(dv.float()).item()
                den = torch.linalg.norm(v.detach().float()).item()
                rel = float(num / (den + 1e-12))

                rels.append((rel, k))
                count += 1
                if count >= max(1, int(getattr(self, "ema_debug_n_params", 5))):
                    break

            if len(rels) == 0:
                logger.info(f"[ema] step={getattr(self, 'global_step', -1)}: no floating params to compare (unexpected)")
                return

            mean_rel = float(sum(r for r, _ in rels) / len(rels))
            max_rel, max_k = max(rels, key=lambda t: t[0])
            logger.info(
                f"[ema] step={getattr(self, 'global_step', -1)}: mean(||θ-θ_ema||/||θ||)={mean_rel:.3e}; "
                f"max={max_rel:.3e} ({max_k})"
            )

        except Exception as e:
            logger.warning(f"[ema] Could not compute theta vs theta_ema diagnostics: {e}")

    def load_checkpoint(self,
                        checkpoint_path,
                        load_ema=False,
                        # If load_ema is True, load the EMA model parameters
                        # If load_ema is False, load the model parameters
                        device=None
                        ):
        '''
            Load a checkpoint from the given path. If load_ema = True and EMA exists, load EMA parameters into self.model
            Also restore the EMA model when enabled.
            Args:
                checkpoint_path: Path to the checkpoint file.
                device: Device to load the checkpoint on. If None, uses the current device.
        '''
        # Check if device is provided, if not, use the current device
        if device is None:
            device = self.device
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        net_sd = checkpoint.get('network_params', None)  # Network state dict
        ema_sd = checkpoint.get('ema_network_params', None)  # EMA state dict if exists
        model_ref = self._model_ref()

        if load_ema and (ema_sd is not None):
            model_ref.load_state_dict(ema_sd)
            if self.is_main_process:
                logger.info(f"→ Loaded EMA model weights into the main model from checkpoint {checkpoint_path}")
        elif net_sd is not None:
            model_ref.load_state_dict(net_sd)
            if self.is_main_process:
                logger.info(f"→ Loaded model weights into the main model from checkpoint {checkpoint_path}")
        else:
            raise KeyError(f"Checkpoint at {checkpoint_path} does not contain 'network_params' or 'ema_network_params'.")
        
        # Load rain-gate parameters if present in checkpoint
        try:
            rg_sd = checkpoint.get('rain_gate_params', None)
            if (rg_sd is not None) and getattr(self, 'rg_enabled', False) and hasattr(self, 'rain_gate') and (self.rain_gate is not None):
                self.rain_gate.load_state_dict(rg_sd)
                if self.is_main_process:
                    logger.info(f"→ Loaded rain-gate head weights from checkpoint {checkpoint_path}")
        except Exception as e:
            if self.is_main_process:
                logger.warning(f"Could not load rain-gate head weights from checkpoint {checkpoint_path}. Error: {e}")
        
        # --- Restore EMA model state if EMA is enabled ---
        if getattr(self, "with_ema", False):
            # Ensure ema_model exists
            if not hasattr(self, "ema_model"):
                self._init_ema()

            if ema_sd is not None:
                # Load EMA weights into the EMA shadow model
                self.ema_model.load_state_dict(ema_sd)
                self.ema_model.eval()
                if self.is_main_process:
                    logger.info(f"→ Restored EMA shadow model weights from checkpoint {checkpoint_path}")
            else:
                # If no EMA weights in checkpoint, sync EMA to current model
                self.ema_model.load_state_dict(model_ref.state_dict())
                self.ema_model.eval()
                if self.is_main_process:
                    logger.info("→ No EMA weights found in checkpoint; synced EMA model to loaded model weights")


    def save_model(self,
                   dirname='./model_params',
                   filename='SBGM.pth'
                   ):
        '''
            Save the model parameters and EMA parameters (if available)
            Args:
                dirname: Directory to save the model parameters.
                filename: Filename to save the model parameters.
        '''
        if not self.is_main_process:
            return None

        os.makedirs(dirname, exist_ok=True)

        model_to_save = self._model_ref()

        state_dicts = {
            'network_params': model_to_save.state_dict(),
            'optimizer_params': self.optimizer.state_dict()
        }
        if getattr(self, 'rg_enabled', False) and hasattr(self, 'rain_gate') and (self.rain_gate is not None):
            state_dicts['rain_gate_params'] = self.rain_gate.state_dict()

        if self.with_ema and hasattr(self, 'ema_model'):
            state_dicts['ema_network_params'] = self.ema_model.state_dict()

        return torch.save(state_dicts, os.path.join(dirname, filename))
    
    def train_batches(self,
              dataloader,
              epochs=10,
              current_epoch=1,
              verbose=True,
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''
        model_ref = self._model_ref()
        self.model.train()
        self._reset_perf_state()

        detect_anomaly = bool(
            self.cfg.get('training', {}).get('detect_anomaly', False)
            or self.cfg.get('debug', {}).get('detect_anomaly', False)
        )

        loss_sum = 0.0
        n_batches = 0

        self.scaler = GradScaler() if torch.cuda.is_available() and use_mixed_precision else None

        disable_pbar = not self.is_main_process
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch", disable=disable_pbar)
        iter_end_t = time.perf_counter()

        for idx, samples in enumerate(pbar):
            batch_t0 = time.perf_counter()
            self._record_perf_time('train_data_wait', batch_t0 - iter_end_t)
            # --- Paper2 sanity check: print LR shapes on first batch of each epoch (main process only)
            if idx == 0 and self.is_main_process:
                try:
                    logger.info("[DEBUG] Batch LR tensor shapes:")
                    for v in self.lr_vars:
                        k_ctx = f"{v}_lr"
                        k_local = f"{v}_lr_local"

                        if k_ctx in samples:
                            logger.info(f"  {k_ctx}: {tuple(samples[k_ctx].shape)}")
                        else:
                            logger.info(f"  {k_ctx}: MISSING")

                        if k_local in samples:
                            logger.info(f"  {k_local}: {tuple(samples[k_local].shape)}")
                        else:
                            logger.info(f"  {k_local}: MISSING")

                except Exception as e:
                    logger.warning(f"[DEBUG] Failed LR shape print: {e}")

            # Samples is a dict with following available keys: 'img', 'y', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            x, y, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)
            lsm_cond = lsm_hr if lsm_hr is not None else lsm

            topo_hr = None
            if isinstance(samples, dict) and ("topo_hr" in samples) and (samples["topo_hr"] is not None):
                topo_hr = samples["topo_hr"].to(self.device)
            topo_cond = topo_hr if topo_hr is not None else topo

            # Build LOCAL LR conditioning (no context channels) for RainGate / baseline logic.
            try:
                local_cond = self._build_local_cond_img(samples)
            except Exception:
                local_cond = cond_images

            # Build final UNet conditioning (may include context features).
            try:
                cond_images = self._build_cond_img(samples)
            except Exception:
                cond_images = local_cond

            self._check_y_runtime(y)

            # === EDM: build lr_ups_baseline if needed ===
            lr_ups_baseline = None
            if self.edm_enabled and self.edm_predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(local_cond)

            # === Rain-gate auxiliary supervision (before CFG dropout affects inputs) ===
            rg_aux_loss = None
            wet_logits = None
            wet_target = None
            if self.rg_enabled and (self.rain_gate is not None):
                gate_inputs = []
                if local_cond is not None:
                    gate_inputs.append(local_cond)
                if self.rg_include_lsm and (lsm_cond is not None):
                    gate_inputs.append(lsm_cond)
                if self.rg_include_topo and (topo_cond is not None):
                    gate_inputs.append(topo_cond)
                if self.rg_include_lr_baseline and (lr_ups_baseline is not None):
                    gate_inputs.append(lr_ups_baseline)

                if len(gate_inputs) > 0:
                    gate_x = torch.cat(gate_inputs, dim=1)
                    wet_logits = self.rain_gate(gate_x)

                    with torch.no_grad():
                        thr = float(self.rg_threshold_mm)
                        bt_hr = None
                        try:
                            if self.back_transforms_train is not None:
                                bt_hr = self.back_transforms_train.get(self.bt_hr_key, None)
                        except Exception:
                            bt_hr = None

                        if callable(bt_hr):
                            x_phys = bt_hr(x)
                            if not isinstance(x_phys, torch.Tensor):
                                x_phys = torch.tensor(x_phys, dtype=torch.float32, device=x.device)
                            wet_target = (x_phys > thr).to(dtype=torch.float32)
                        else:
                            if self.is_main_process and idx == 0:
                                logger.warning("[rain_gate] back_transforms_train missing or invalid; using model-space thresholding for rain gate target.")
                            wet_target = (x > self.rg_threshold_modelSpace).to(dtype=torch.float32)

                        if wet_target.shape[1] != 1:
                            wet_target = wet_target[:, :1, :, :]

                    pos_w = torch.tensor(self.rg_pos_weight, device=x.device, dtype=torch.float32)
                    wet_logits_use = wet_logits
                    if (wet_logits_use is not None) and (wet_target is not None):
                        if wet_logits_use.shape[-2:] != wet_target.shape[-2:]:
                            wet_logits_use = F.interpolate(
                                wet_logits_use,
                                size=wet_target.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                    bce = F.binary_cross_entropy_with_logits(wet_logits_use, wet_target, pos_weight=pos_w)
                    wet_logits = wet_logits_use
                    rg_aux_loss = bce * self.rg_loss_weight

            # === Optional: Use gate to reweight main loss on wet pixels (with warm start and ramp) ===
            pixel_weight_map = None
            if self.rg_enabled and (self.rain_gate is not None):
                do_reweight = self.rg_reweight_enabled and (current_epoch > self.rg_warm_start)
                if do_reweight and (wet_logits is not None):
                    rg_cfg = self.cfg.get('rain_gate', {})
                    p = torch.sigmoid(wet_logits)
                    if p.shape[-2:] != x.shape[-2:]:
                        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    strategy = str(rg_cfg.get('weight_strategy', 'prob')).lower()
                    alpha = float(rg_cfg.get('weight_alpha', 2.0))
                    clip_max = float(rg_cfg.get('clip_max', 5.0))
                    detach_w = bool(rg_cfg.get('detach_weights', True))

                    if strategy == 'binary':
                        thr_p = float(rg_cfg.get('binary_threshold', 0.5))
                        w_core = 1.0 + alpha * (p >= thr_p).to(dtype=p.dtype)
                    else:
                        gamma = float(rg_cfg.get('prob_gamma', 1.0))
                        w_core = 1.0 + alpha * (p.clamp(0, 1) ** gamma)

                    if self.rg_ramp > 0:
                        phase = min(1.0, max(0.0, (current_epoch - self.rg_warm_start) / max(1, self.rg_ramp)))
                        ramp_prog = 0.5 * (1 - math.cos(math.pi * phase))
                    else:
                        ramp_prog = 1.0

                    w = 1.0 + (w_core - 1.0) * ramp_prog
                    w = w.clamp(min=1.0, max=clip_max)
                    if detach_w:
                        w = w.detach()
                    pixel_weight_map = w

            # Diagnostics / asserts
            self._assert_all_finite('x', x)
            self._assert_all_finite('cond_images', cond_images)
            self._assert_all_finite('lr_ups_baseline', lr_ups_baseline)

            cfg_diagnostics = self.cfg.get("diagnostics", {})
            do_log = bool(cfg_diagnostics.get("per_batch_stats", False))
            every = int(cfg_diagnostics.get("log_every", 100))

            rg_cfg = self.cfg.get("rain_gate", {})
            save_rg_train_viz = bool(rg_cfg.get("save_train_viz", False))
            viz_every_n_epochs = int(rg_cfg.get("viz_every_n_epochs", 10) or 10)
            viz_first_batch_only = bool(rg_cfg.get("viz_first_batch_only", True))

            save_rg_train_reliability = bool(rg_cfg.get("save_train_reliability", False))
            reliability_every_n_epochs = int(rg_cfg.get("reliability_every_n_epochs", viz_every_n_epochs) or viz_every_n_epochs)
            reliability_first_batch_only = bool(rg_cfg.get("reliability_first_batch_only", True))

            should_save_rg_train_viz = (
                save_rg_train_viz
                and (wet_logits is not None)
                and (viz_every_n_epochs > 0)
                and (current_epoch % viz_every_n_epochs == 0)
                and ((idx == 0) if viz_first_batch_only else True)
            )

            if should_save_rg_train_viz:
                _save_weight_map_viz(
                    weight_map=pixel_weight_map if pixel_weight_map is not None else torch.sigmoid(wet_logits),
                    wet_probs=torch.sigmoid(wet_logits),
                    wet_target=wet_target,
                    epoch=current_epoch,
                    step=idx,
                    prefix='train',
                    save_path=self.path_diagnostics,
                )
            should_save_rg_train_reliability = (
                save_rg_train_reliability
                and (wet_logits is not None)
                and (wet_target is not None)
                and (reliability_every_n_epochs > 0)
                and (current_epoch % reliability_every_n_epochs == 0)
                and ((idx == 0) if reliability_first_batch_only else True)
            )

            if should_save_rg_train_reliability:
                try:
                    _plot_reliability_curve(
                        probs=torch.sigmoid(wet_logits).detach(),
                        targets=wet_target.detach(),
                        epoch=current_epoch,
                        prefix='train',
                        save_path=self.path_diagnostics,
                    )
                except Exception as e:
                    logger.warning(f"[rain_gate][train] Could not save reliability curve at epoch {current_epoch}, batch {idx}. Error: {e}")

            hr = x
            lr_hr = lr_ups_baseline

            lr_lr = None
            if cond_images is not None:
                try:
                    cmap = getattr(self, "cond_channel_map_cfg", None)
                    main_lr_cond = None if cmap is None else cmap.get("main_lr_cond", None)

                    if (cmap is not None) and (main_lr_cond is not None) and (main_lr_cond in cmap["slices"]):
                        dual_lr = bool(self.cfg.get("lowres", {}).get("dual_lr", False))
                        kind = "main"
                        if dual_lr and ("lr_only" in cmap["slices"][main_lr_cond]):
                            try:
                                plot_ch = int(self.cfg.get("visualization", {}).get("plot_dual_lr_channel", 0))
                            except Exception:
                                plot_ch = 0
                            kind = "lr_only" if plot_ch == 1 else "main"

                        s0, s1 = self._cond_slice(main_lr_cond, kind=kind)
                        if cond_images.shape[1] >= s1:
                            lr_lr = cond_images[:, s0:s1, :, :]
                except Exception as e:
                    if self.is_main_process and (idx == 0) and (current_epoch == 1):
                        logger.warning(f"[train] Could not extract lr_lr diagnostic slice from cond_images. Error: {e}")
                    lr_lr = None

            residual = hr - lr_hr if (hr is not None and lr_hr is not None) else None
            if self.is_main_process and (idx == 0) and (current_epoch == 1) and (self.edm_enabled and not self.edm_predict_residual):
                logger.info("[train] edm.predict_residual=False → lr_ups_baseline not built; residual diagnostics will be None (expected).")

            if do_log and (idx % every == 0):
                tensor_stats(hr, "train/hr_norm")
                if lr_hr is not None:
                    tensor_stats(lr_hr, "train/lr_hr_norm")
                if residual is not None:
                    tensor_stats(residual, "train/residual_hr_space")

            clamp_warn = float(cfg_diagnostics.get("warn_if_abs_gt", 15.0))
            if do_log and (idx % every == 0) and (clamp_warn > 0.0) and (residual is not None):
                mx = float(residual.abs().amax().item())
                if mx > clamp_warn:
                    logger.warning(f"[diagnostics][train] Batch {idx}: |residual| max {mx:.2f} exceeds warn_if_abs_gt {clamp_warn}. Consider residual normalization, tail clamp or loss robustification.")

            cond_images, lsm_cond, topo_cond, y, lr_ups_baseline, drop_info = apply_cfg_dropout(
                cond_images=cond_images,
                lsm_cond=lsm_cond,
                topo_cond=topo_cond,
                y=y,
                lr_ups=lr_ups_baseline,
                cfg_guidance=self.cfg_guidance,
            )

            self._check_y_runtime(y)
            self.optimizer.zero_grad(set_to_none=True)

            for name, tensor in zip(['x', 'y', 'cond_images', 'lsm', 'topo'], [x, y, cond_images, lsm, topo]):
                if tensor is not None:
                    assert tensor.device == x.device, f"{name} is on device {tensor.device}, expected {x.device}"

            t_forward0 = time.perf_counter()
            if detect_anomaly:
                with torch.autograd.set_detect_anomaly(True):
                    if self.scaler is not None:
                        with autocast():
                            batch_loss = self.loss_fn(model_ref,
                                                      x,
                                                      y=y,
                                                      cond_img=cond_images,
                                                      lsm_cond=lsm_cond,
                                                      topo_cond=topo_cond,
                                                      sdf_cond=sdf,
                                                      lr_ups=lr_ups_baseline,
                                                      pixel_weight_map=pixel_weight_map)
                    else:
                        batch_loss = self.loss_fn(model_ref,
                                                  x,
                                                  y=y,
                                                  cond_img=cond_images,
                                                  lsm_cond=lsm_cond,
                                                  topo_cond=topo_cond,
                                                  sdf_cond=sdf,
                                                  lr_ups=lr_ups_baseline,
                                                  pixel_weight_map=pixel_weight_map)
            else:
                if self.scaler is not None:
                    with autocast():
                        batch_loss = self.loss_fn(model_ref,
                                                  x,
                                                  y=y,
                                                  cond_img=cond_images,
                                                  lsm_cond=lsm_cond,
                                                  topo_cond=topo_cond,
                                                  sdf_cond=sdf,
                                                  lr_ups=lr_ups_baseline,
                                                  pixel_weight_map=pixel_weight_map)
                else:
                    batch_loss = self.loss_fn(model_ref,
                                              x,
                                              y=y,
                                              cond_img=cond_images,
                                              lsm_cond=lsm_cond,
                                              topo_cond=topo_cond,
                                              sdf_cond=sdf,
                                              lr_ups=lr_ups_baseline,
                                              pixel_weight_map=pixel_weight_map)
            self._record_perf_time('train_forward', time.perf_counter() - t_forward0)

            if rg_aux_loss is not None:
                batch_loss = batch_loss + rg_aux_loss
            self._assert_all_finite('batch_loss', batch_loss)

            # # === In-loop monitoring (lightweight): cosine and HR-LR correlation ===
            # # IMPORTANT: run this under inference_mode so we do not create a second autograd
            # # graph on the live training model before backward on batch_loss.
            # monitor_cfg = self.cfg.get('monitoring', {})
            # log_every = monitor_cfg.get('edm_metrics_every', 50)
            # try:
            #     epoch_len = len(dataloader)
            # except Exception:
            #     epoch_len = None

            # if epoch_len is not None:
            #     global_step = (current_epoch - 1) * epoch_len + idx
            # else:
            #     global_step = int(getattr(self, "global_step", 0))
            # edm_on = self.cfg.get('edm', {}).get('enabled', False)

            # if self.is_main_process and edm_on and log_every > 0 and (global_step % log_every == 0):
            #     # IMPORTANT: temporarily switch to eval mode for metric probing so we do not
            #     # mutate any train-mode buffers/state on the live model before backward.
            #     was_training = self.model.training
            #     self.model.eval()
            #     try:
            #         with torch.inference_mode():
            #             metrics = in_loop_metrics(
            #                 loss_obj=self.loss_fn,
            #                 model=self.model,
            #                 x0=x,
            #                 y=y,
            #                 cond_img=cond_images,
            #                 lsm_cond=lsm_cond,
            #                 topo_cond=topo_cond,
            #                 lr_ups=lr_ups_baseline,
            #                 eval_land_only=self.eval_land_only,
            #             )
            #     finally:
            #         if was_training:
            #             self.model.train()

            #     self.live_metrics['steps'].append(global_step)
            #     self.live_metrics['edm_cosine'].append(float(metrics.get('edm_cosine', float('nan'))))
            #     self.live_metrics['hr_lr_corr'].append(float(metrics.get('hr_lr_corr', float('nan'))))
            t_backward0 = time.perf_counter()
            if detect_anomaly:
                with torch.autograd.set_detect_anomaly(True):
                    if self.scaler is not None:
                        self.scaler.scale(batch_loss).backward()
                    else:
                        batch_loss.backward()
            else:
                if self.scaler is not None:
                    self.scaler.scale(batch_loss).backward()
                else:
                    batch_loss.backward()
            self._record_perf_time('train_backward', time.perf_counter() - t_backward0)

            t_step0 = time.perf_counter()
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self._record_perf_time('train_step', time.perf_counter() - t_step0)

            self.global_step += 1
            if self.with_ema:
                if not hasattr(self, "ema_model"):
                    self._init_ema()

                if self.global_step >= self.ema_warmup_steps:
                    self._update_ema()
                else:
                    self.ema_model.load_state_dict(model_ref.state_dict())

            if self.with_ema and (self.ema_debug_every > 0) and (self.global_step % self.ema_debug_every == 0):
                self._log_ema_delta()

            monitor_cfg = self.cfg.get('monitoring', {})
            log_every = monitor_cfg.get('edm_metrics_every', 50)
            try:
                epoch_len = len(dataloader)
            except Exception:
                epoch_len = None

            if epoch_len is not None:
                global_step = (current_epoch - 1) * epoch_len + idx
            else:
                global_step = int(getattr(self, "global_step", 0))
            edm_on = self.cfg.get('edm', {}).get('enabled', False)

            if self.is_main_process and edm_on and log_every > 0 and (global_step % log_every == 0):
                try:
                    with torch.inference_mode():
                        metrics = in_loop_metrics(
                            loss_obj=self.loss_fn,
                            model=model_ref,
                            x0=x,
                            y=y,
                            cond_img=cond_images,
                            lsm_cond=lsm_cond,
                            topo_cond=topo_cond,
                            lr_ups=lr_ups_baseline,
                            eval_land_only=self.eval_land_only,
                        )
                    if metrics is not None:
                        self.live_metrics['steps'].append(global_step)
                        self.live_metrics['edm_cosine'].append(float(metrics.get('edm_cosine', float('nan'))))
                        self.live_metrics['hr_lr_corr'].append(float(metrics.get('hr_lr_corr', float('nan'))))
                except Exception as e:
                    logger.warning(f"[monitor][train] Could not compute in-loop metrics at step {global_step}. Error: {e}")

            loss_sum += batch_loss.item()
            n_batches += 1
            if self.is_main_process and (idx % self.cfg['training'].get('train_postfix_every', 10) == 0):
                pbar.set_postfix(loss=loss_sum / max(1, n_batches), rg_bce=float(rg_aux_loss.item()) if rg_aux_loss is not None else None)

            iter_end_t = time.perf_counter()
            self._record_perf_time('train_batch', iter_end_t - batch_t0)

        avg_loss = loss_sum / max(1, n_batches)

        if verbose and self.is_main_process:
            logger.info(f"→ Epoch {getattr(self, 'epoch', '?')} completed: Avg. training Loss: {avg_loss:.4f}")
            if self._timer_enabled():
                self.log_perf_summary(prefix=f"[perf][train][epoch={current_epoch}]")

        return avg_loss

    def train(self,
              train_dataloader,
              val_dataloader,
              gen_dataloader,
              cfg,
              epochs=1,
              verbose=True,
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                train_dataloader: Dataloader to run through.
                val_dataloader: Dataloader to run through for validation.
                epochs: Number of epochs to train for.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''

        if self.is_main_process:
            logger.info(f"→ Classifier-Free Guidance (CFG) enabled: {self.cfg_guidance.get('enabled', False)}")
            if self.cfg_guidance.get('enabled', False):
                logger.info(f"   ▸ Dropout probability for LR conditions: {self.cfg_guidance.get('drop_prob_lr', 0.1)}")
                logger.info(f"   ▸ Dropout probability for static geo: {self.cfg_guidance.get('drop_prob_geo', self.cfg_guidance.get('drop_prob', 0.1))}")

            logger.info(f"→ EMA enabled: {self.with_ema}; decay: {getattr(self, 'ema_decay', None)}; eval_use_ema: {cfg['training'].get('eval_use_ema', True)}")

        train_losses = []
        val_losses = []

        train_loss = float('inf')
        val_loss = float('inf')
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            self.epoch = epoch

            if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
                train_dataloader.sampler.set_epoch(epoch)

            if verbose and self.is_main_process:
                logger.info(f"\n\n      ▸ Starting epoch {epoch}/{epochs}...")

            train_loss = self.train_batches(train_dataloader,
                                            epochs=epochs,
                                            current_epoch=epoch,
                                            verbose=verbose,
                                            use_mixed_precision=use_mixed_precision)

            train_losses.append(train_loss)

            self._dist_barrier()
            if self.is_main_process:
                val_loss = self.validate_batches(val_dataloader,
                                                 epochs=epochs,
                                                 current_epoch=epoch,
                                                 verbose=verbose)
                val_losses.append(val_loss)
            self._dist_barrier()

            if self.is_main_process:
                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(val_loss)
                    else:
                        self.lr_scheduler.step()
                    if verbose:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        logger.info(f"→ Learning rate after epoch {epoch}: {current_lr:.6f}")

                improved = val_loss < best_loss

                if improved:
                    best_loss = val_loss
                    self.save_model(dirname=self.checkpoint_dir, filename=self.checkpoint_name)
                    logger.info(f"→ Best model saved with validation loss: {best_loss:.4f} at epoch {epoch}.")
                    logger.info(f"→ Checkpoint saved to {os.path.join(self.checkpoint_dir, self.checkpoint_name)}")

                losses = {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                with open(os.path.join(self.path_losses, 'losses' + f'_{self.model_string}.pkl'), 'wb') as f:
                    pickle.dump(losses, f)

                if cfg['visualization']['create_figs'] and cfg['visualization']['plot_losses']:
                    self.plot_losses(train_losses,
                                     val_losses=val_losses,
                                     save_path=self.path_figures,
                                     save_name=f'losses_plot_{self.model_string}.png',
                                     show_plot=cfg['visualization']['show_figs'])

                if (self.monitor_plot_every_n_epochs > 0) and (epoch % self.monitor_plot_every_n_epochs == 0):
                    try:
                        self._plot_live_metrics(self.path_metrics, n_samples=cfg['data_handling']['n_gen_samples'])
                    except Exception as e:
                        logger.warning(f"[monitor] Could not plot live metrics at epoch {epoch}. Error: {e}")

                if cfg['visualization']['create_figs'] and cfg['data_handling']['n_gen_samples'] > 0:
                    gen_every_n_epochs = int(cfg['visualization'].get('gen_and_plot_every_n_epochs', 1) or 1)
                    if gen_every_n_epochs < 1:
                        gen_every_n_epochs = 1

                    on_schedule = (epoch % gen_every_n_epochs == 0)
                    do_gen_plot = bool(improved or on_schedule)
                    allow_gen_plot_in_ddp = bool(cfg.get('visualization', {}).get('allow_gen_plot_in_ddp', False))
                    do_light_monitor = bool(cfg.get('visualization', {}).get('train_monitor_generate', False))
                    light_monitor_every = int(cfg.get('visualization', {}).get('train_monitor_every_n_epochs', gen_every_n_epochs) or gen_every_n_epochs)
                    do_light_monitor_now = do_light_monitor and (epoch % max(1, light_monitor_every) == 0)

                    if do_gen_plot:
                        if self.distributed and not allow_gen_plot_in_ddp:
                            logger.info(
                                f"→ Skipping generate_and_plot_samples at epoch {epoch} because distributed training is active. "
                                f"Set visualization.allow_gen_plot_in_ddp=true to override."
                            )
                        else:
                            if on_schedule:
                                logger.info(f"→ Generating and plotting samples at epoch {epoch} (every {gen_every_n_epochs} epochs)...")
                            if improved:
                                logger.info(f"→ Generating and plotting samples at epoch {epoch} (new best model)...")
                            self.generate_and_plot_samples(gen_dataloader,
                                                           cfg=cfg,
                                                           epoch=epoch)

                    if do_light_monitor_now:
                        try:
                            self.generate_training_monitor_samples(gen_dataloader=gen_dataloader, cfg=cfg, epoch=epoch)
                        except Exception as e:
                            logger.warning(f"[monitor] Could not generate lightweight training monitor samples at epoch {epoch}. Error: {e}")

                logger.info(f"→ Epoch {epoch}/{epochs} completed. \n\n")

        return train_loss, val_loss

    def validate_batches(self,
                    dataloader,
                    epochs=1,
                    current_epoch=1,
                    verbose=True
                 ):
        '''
            Method to run through the validation batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
        '''

        model_ref = self._model_ref()
        self.model.eval()
        edm_on = bool(self.cfg.get('edm', {}).get('enabled', False))

        use_ema_for_val = bool(self.cfg['training'].get('eval_use_ema', True))
        if self.with_ema and use_ema_for_val and hasattr(self, 'ema_model'):
            self.ema_model.eval()
            model_eval = self.ema_model
        else:
            model_eval = model_ref

        loss = 0.0
        n_batches = 0
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {current_epoch}/{epochs}", unit="batch", disable=not self.is_main_process)

        for idx, samples in enumerate(pbar):
            x, y, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)
            lsm_cond = lsm_hr if lsm_hr is not None else lsm

            topo_hr = None
            if isinstance(samples, dict) and ("topo_hr" in samples) and (samples["topo_hr"] is not None):
                topo_hr = samples["topo_hr"].to(self.device)
            topo_cond = topo_hr if topo_hr is not None else topo

            try:
                local_cond = self._build_local_cond_img(samples)
            except Exception:
                local_cond = cond_images

            try:
                cond_images = self._build_cond_img(samples)
            except Exception:
                cond_images = local_cond

            self._check_y_runtime(y)

            lr_ups_baseline = None
            if edm_on and self.edm_predict_residual:
                lr_ups_baseline = self._build_lr_ups_baseline(local_cond)

            if self.is_main_process and (idx == 0) and (current_epoch == 1) and edm_on and (not self.edm_predict_residual):
                logger.info("[val] edm.predict_residual=False → lr_ups_baseline not built (expected).")

            pixel_weight_map = None
            wet_logits_val = None
            wet_target = None
            rg_val_bce = None
            rg_val_bce_t = None

            if getattr(self, 'rg_enabled', False) and (getattr(self, 'rain_gate', None) is not None):
                gate_inputs = []
                if local_cond is not None:
                    gate_inputs.append(local_cond)
                if self.rg_include_lsm and (lsm_cond is not None):
                    gate_inputs.append(lsm_cond)
                if self.rg_include_topo and (topo_cond is not None):
                    gate_inputs.append(topo_cond)
                if self.rg_include_lr_baseline and (lr_ups_baseline is not None):
                    gate_inputs.append(lr_ups_baseline)

                if len(gate_inputs) > 0 and self.rain_gate is not None:
                    gate_x = torch.cat(gate_inputs, dim=1)
                    with torch.no_grad():
                        wet_logits_val = self.rain_gate(gate_x)
                        p = torch.sigmoid(wet_logits_val)
                        if p.shape[-2:] != x.shape[-2:]:
                            p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    wet_logits_val = None

                do_reweight = bool(self.cfg.get('rain_gate', {}).get('reweight_enabled', False)) and (current_epoch > self.rg_warm_start)
                if do_reweight and (wet_logits_val is not None):
                    p = torch.sigmoid(wet_logits_val)
                    if p.shape[-2:] != x.shape[-2:]:
                        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    rg_cfg = self.cfg.get('rain_gate', {})
                    strategy = str(rg_cfg.get('weight_strategy', 'prob')).lower()
                    alpha = float(rg_cfg.get('weight_alpha', 2.0))
                    clip_max = float(rg_cfg.get('clip_max', 5.0))

                    if strategy == 'binary':
                        thr_p = float(rg_cfg.get('binary_threshold', 0.5))
                        core = (p >= thr_p).to(dtype=p.dtype)
                    else:
                        gamma = float(rg_cfg.get('prob_gamma', 1.0))
                        core = (p.clamp(0,1) ** gamma)

                    if self.rg_ramp > 0:
                        phase = min(1.0, max(0.0, (current_epoch - self.rg_warm_start) / max(1, self.rg_ramp)))
                        ramp_prog = 0.5 * (1 - math.cos(math.pi * phase))
                    else:
                        ramp_prog = 1.0

                    w = 1.0 + (((1.0 + alpha * core) - 1.0) * ramp_prog)
                    pixel_weight_map = w.clamp(min=1.0, max=clip_max).detach()

            if wet_logits_val is not None:
                try:
                    with torch.no_grad():
                        thr = float(self.rg_threshold_mm)
                        bt_hr = self.back_transforms_train.get(self.bt_hr_key, None) if self.back_transforms_train is not None else None
                        if callable(bt_hr):
                            x_phys = bt_hr(x)
                            if not isinstance(x_phys, torch.Tensor):
                                x_phys = torch.tensor(x_phys, dtype=torch.float32, device=x.device)
                            wet_target = (x_phys > thr).to(dtype=torch.float32)
                        else:
                            wet_target = (x > self.rg_threshold_modelSpace).to(dtype=torch.float32)
                        if wet_target.shape[1] != 1:
                            wet_target = wet_target[:, :1, :, :]

                    pos_w = torch.tensor(self.rg_pos_weight, device=x.device, dtype=torch.float32)
                    wet_logits_use = wet_logits_val
                    if (wet_logits_use is not None) and (wet_target is not None):
                        if wet_logits_use.shape[-2:] != wet_target.shape[-2:]:
                            wet_logits_use = F.interpolate(
                                wet_logits_use,
                                size=wet_target.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                    bce = F.binary_cross_entropy_with_logits(wet_logits_use, wet_target, pos_weight=pos_w)
                    rg_val_bce_t = bce
                    rg_val_bce = float(bce.item())
                    wet_logits_val = wet_logits_use
                except Exception as e:
                    if self.is_main_process:
                        logger.warning(f"[rain_gate] Could not compute validation BCE or target. Error: {e}")
                    wet_target = None
                    rg_val_bce_t = None
                    rg_val_bce = None

            with torch.inference_mode():
                if hasattr(self, 'scaler') and self.scaler:
                    with autocast():
                        batch_loss = self.loss_fn(model_eval,
                                                  x,
                                                  y=y,
                                                  cond_img=cond_images,
                                                  lsm_cond=lsm_cond,
                                                  topo_cond=topo_cond,
                                                  sdf_cond=sdf,
                                                  lr_ups=lr_ups_baseline,
                                                  pixel_weight_map=pixel_weight_map
                                                  )
                else:
                    batch_loss = self.loss_fn(model_eval,
                                              x,
                                              y=y,
                                              cond_img=cond_images,
                                              lsm_cond=lsm_cond,
                                              topo_cond=topo_cond,
                                              sdf_cond=sdf,
                                              lr_ups=lr_ups_baseline,
                                              pixel_weight_map=pixel_weight_map
                                              )

                if (getattr(self, 'rg_enabled', False) and (rg_val_bce_t is not None) and (self.rg_loss_weight > 0.0)):
                    batch_loss = batch_loss + rg_val_bce_t * self.rg_loss_weight

                monitor_cfg = self.cfg.get('monitoring', {})
                log_every = monitor_cfg.get('edm_metrics_every', 50)
                if self.is_main_process and edm_on and log_every > 0 and (idx % log_every == 0):
                    metrics = in_loop_metrics(
                        loss_obj=self.loss_fn,
                        model=model_eval,
                        x0=x,
                        y=y,
                        cond_img=cond_images,
                        lsm_cond=lsm_cond,
                        topo_cond=topo_cond,
                        lr_ups=lr_ups_baseline,
                        eval_land_only=self.eval_land_only,
                    )
                    if verbose and metrics is not None:
                        logger.info(f"→ [monitor][val] Step {idx}: EDM cosine metric: {metrics.get('edm_cosine', float('nan')):.4f}")
                        logger.info(f"→ [monitor][val] Step {idx}: HR-LR corr: {metrics.get('hr_lr_corr', float('nan')):.4f}")

            loss += batch_loss.item()
            n_batches += 1
            if self.is_main_process and (idx % self.cfg['training'].get('train_postfix_every', 10) == 0):
                pbar.set_postfix(loss=loss / max(1, n_batches), rg_bce=rg_val_bce if wet_logits_val is not None else None)

        avg_loss = loss / max(1, n_batches)

        if verbose and self.is_main_process:
            logger.info(f'→ Validation Loss: {avg_loss:.4f}')

        return avg_loss
    
    def generate_training_monitor_samples(self,
                                          gen_dataloader,
                                          cfg,
                                          epoch):
        """
        Lightweight qualitative monitor intended for DDP-safe rank-0 use.
        Generates a single case with a small ensemble and plots:
          local LR, HR truth, 4 generated members, ensemble mean + extrema summary.
        """
        if not self.is_main_process:
            return

        model_ref = self._model_ref()
        was_training = model_ref.training
        model_ref.eval()

        if hasattr(gen_dataloader, 'sampler') and hasattr(gen_dataloader.sampler, 'set_epoch'):
            try:
                gen_dataloader.sampler.set_epoch(epoch)
            except Exception:
                pass
        batch = next(iter(gen_dataloader))
        x, y, cond_images, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(batch, self.device)

        topo_hr = None
        if isinstance(batch, dict) and ('topo_hr' in batch) and (batch['topo_hr'] is not None):
            topo_hr = batch['topo_hr'].to(self.device)
        topo_cond = topo_hr if topo_hr is not None else topo
        lsm_cond = lsm_hr if lsm_hr is not None else lsm

        try:
            local_cond = self._build_local_cond_img(batch)
        except Exception:
            local_cond = cond_images

        try:
            cond_images = self._build_cond_img(batch)
        except Exception:
            cond_images = local_cond

        lr_ups_baseline = None
        if bool(cfg.get('edm', {}).get('enabled', False)) and self.edm_predict_residual:
            lr_ups_baseline = self._build_lr_ups_baseline(local_cond)

        member_count = int(cfg.get('visualization', {}).get('train_monitor_n_members', 4) or 4)
        member_count = max(1, member_count)

        edm_on = bool(cfg.get('edm', {}).get('enabled', False))
        if edm_on:
            sampler_fn = edm_sampler
        else:
            sampler_type = cfg['sampler']['sampler_type']
            if sampler_type == 'pc_sampler':
                sampler_fn = pc_sampler
            elif sampler_type == 'Euler_Maruyama_sampler':
                sampler_fn = Euler_Maruyama_sampler
            elif sampler_type == 'ode_sampler':
                sampler_fn = ode_sampler
            else:
                raise ValueError(f"Sampler type {sampler_type} not recognized.")

        sample0 = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                sample0[k] = v[0].detach().cpu()
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                sample0[k] = v[0]
            else:
                sample0[k] = v

        full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        crop_region_hr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx")

        paper2 = (cfg.get('paper2', {}) or {})
        spatial = (paper2.get('spatial_context', {}) or {})
        spatial_mode = str(spatial.get('mode', '')).lower()
        if spatial_mode == 'large_domain' and (self.full_domain_dims_lr is not None):
            lr_crop_bounds_eff = [0, self.full_domain_dims_lr[1], 0, self.full_domain_dims_lr[0]]
        else:
            lr_crop_bounds_eff = self.crop_region_lr
        crop_region_lr_str = crop_bounds_to_stats_str(lr_crop_bounds_eff, order="xxyy")

        back_transforms = build_back_transforms_from_stats(
            hr_var=cfg['highres']['variable'],
            hr_model=cfg['highres']['model'],
            domain_str_hr=full_domain_dims_str_hr,
            crop_region_str_hr=crop_region_hr_str,
            hr_scaling_method=cfg['highres']['scaling_method'],
            hr_buffer_frac=cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
            lr_vars=cfg['lowres']['condition_variables'],
            lr_model=cfg['lowres']['model'],
            domain_str_lr=full_domain_dims_str_lr,
            crop_region_str_lr=crop_region_lr_str,
            lr_scaling_methods=cfg['lowres']['scaling_methods'],
            lr_buffer_frac=cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
            scaling_split=str(cfg.get('transforms', {}).get('scaling_split', 'train')),
            stats_load_dir=cfg['data_handling']['stats_load_dir'],
            use_dual_lr=bool(cfg.get('lowres', {}).get('dual_lr', False)),
            lr_main_var_scale=str(cfg.get('lowres', {}).get('lr_main_var_scale', 'LR')),
            use_hrspace_for_non_main=bool(cfg.get('lowres', {}).get('use_hrspace_for_non_main', False)),
            main_condition=cfg.get('lowres', {}).get('main_condition_variable', None),
        )

        member_outputs = []
        with torch.inference_mode():
            for _ in range(member_count):
                if edm_on:
                    gen_out = sampler_fn(
                        model=model_ref,
                        shape=x[:1].shape,
                        y=y[:1] if y is not None else None,
                        cond_img=cond_images[:1] if cond_images is not None else None,
                        lsm_cond=lsm_cond[:1] if lsm_cond is not None else None,
                        topo_cond=topo_cond[:1] if topo_cond is not None else None,
                        lr_ups=lr_ups_baseline[:1] if lr_ups_baseline is not None else None,
                        cfg=cfg,
                    )
                else:
                    gen_out = sampler_fn(
                        model=model_ref,
                        marginal_prob_std_fn=self.marginal_prob_std_fn,
                        diffusion_coeff_fn=self.diffusion_coeff_fn,
                        batch_size=1,
                        device=self.device,
                        y=y[:1] if y is not None else None,
                        condition=cond_images[:1] if cond_images is not None else None,
                        lsm=lsm_cond[:1] if lsm_cond is not None else None,
                        topo=topo_cond[:1] if topo_cond is not None else None,
                    )

                if isinstance(gen_out, tuple):
                    gen_out = gen_out[0]
                member_outputs.append(gen_out[:1].detach().cpu())

        generated_members = torch.cat(member_outputs, dim=0)

        date_value = sample0.get('date', None)
        if isinstance(date_value, (list, tuple)) and len(date_value) > 0:
            date_value = date_value[0]
        if torch.is_tensor(date_value):
            date_value = date_value.item()
        if date_value is not None:
            date_value = str(date_value)

        fig, _ = plot_training_monitor_generated(
            sample0,
            generated_members,
            cfg,
            date=date_value,
            transform_back_bf_plot=True,
            back_transforms=back_transforms,
            figsize=tuple(cfg.get('visualization', {}).get('train_monitor_figsize', (16, 4.8))),
        )

        save_dir = os.path.join(self.path_figures, 'training_monitor')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'train_monitor_epoch_{epoch:03d}.png')
        fig.savefig(save_path, bbox_inches='tight', dpi=250)
        plt.close(fig)
        logger.info(f"[monitor] Saved lightweight training monitor figure to {save_path}")

        if was_training:
            model_ref.train()

    def generate_and_plot_samples(self,
                            gen_dataloader,
                            cfg,
                            epoch,
                          ):
        # This method is intended to run on the main process only.
        if not self.is_main_process:
            return

        if self.distributed and not bool(cfg.get('visualization', {}).get('allow_gen_plot_in_ddp', False)):
            logger.info("→ Skipping sample generation/plotting on main process because distributed training is active.")
            return

        model_ref = self._model_ref()
        model_sd_backup = copy.deepcopy(model_ref.state_dict())

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net_sd = checkpoint.get('network_params', None)
        ema_sd = checkpoint.get('ema_network_params', None)
        use_ema_for_gen = bool(cfg['training'].get('eval_use_ema', True))

        if self.with_ema and use_ema_for_gen and (ema_sd is not None):
            model_ref.load_state_dict(ema_sd)
            logger.info(f"→ Loaded EMA model weights into the main model from checkpoint {self.checkpoint_path} for sampling.")
        elif net_sd is not None:
            model_ref.load_state_dict(net_sd)
            logger.info(f"→ Loaded model weights into the main model from checkpoint {self.checkpoint_path} for sampling.")
        else:
            logger.warning(f"→ No EMA weights in checkpoint; using current in-memory network weights for sampling.")

        if self.with_ema:
            if not hasattr(self, 'ema_model'):
                self._init_ema()  # Initialize EMA if not already
            if ema_sd is not None:
                self.ema_model.load_state_dict(ema_sd)  # Sync EMA model

        model_ref.eval()

        edm_on = bool((cfg.get('edm', {}).get('enabled', False)))
        if edm_on:
            sampler_edm = edm_sampler
            sampler = None
            logger.info("→ Sampling using EDM sampler...")
        else:
            sampler_edm = None
            if cfg['sampler']['sampler_type'] == 'pc_sampler':
                sampler = pc_sampler
            elif cfg['sampler']['sampler_type'] == 'Euler_Maruyama_sampler':
                sampler = Euler_Maruyama_sampler
            elif cfg['sampler']['sampler_type'] == 'ode_sampler':
                sampler = ode_sampler
            else:
                raise ValueError(f"Sampler type {cfg['sampler']['sampler_type']} not recognized. Please choose from 'pc_sampler', 'Euler_Maruyama_sampler', or 'ode_sampler'.")

        full_domain_dims_str_hr = f"{self.full_domain_dims_hr[0]}x{self.full_domain_dims_hr[1]}" if self.full_domain_dims_hr is not None else "full_domain"
        full_domain_dims_str_lr = f"{self.full_domain_dims_lr[0]}x{self.full_domain_dims_lr[1]}" if self.full_domain_dims_lr is not None else "full_domain"
        crop_region_hr_str = crop_bounds_to_stats_str(self.crop_region_hr, order="yyxx")

        paper2 = (cfg.get('paper2', {}) or {})
        spatial = (paper2.get('spatial_context', {}) or {})
        spatial_mode = str(spatial.get('mode', '')).lower()
        if spatial_mode == 'large_domain' and (self.full_domain_dims_lr is not None):
            lr_crop_bounds_eff = [0, self.full_domain_dims_lr[1], 0, self.full_domain_dims_lr[0]]
        else:
            lr_crop_bounds_eff = self.crop_region_lr

        crop_region_lr_str = crop_bounds_to_stats_str(lr_crop_bounds_eff, order="xxyy")

        scaling_split = str(cfg.get('transforms', {}).get('scaling_split', 'train'))
        if epoch == 1:
            try:
                logger.info(f"[gen] Using scaling_split='{scaling_split}' for back-transforms.")
                logger.info(f"[gen] HR stats crop string: {crop_region_hr_str}")
                logger.info(f"[gen] LR stats crop string: {crop_region_lr_str}")
                if hasattr(self, "cond_channel_map_cfg"):
                    logger.info(f"[gen] LR channel map (cfg mirror): {self.cond_channel_map_cfg}")
            except Exception:
                pass

        back_transforms = build_back_transforms_from_stats(
                            hr_var              = cfg['highres']['variable'],
                            hr_model            = cfg['highres']['model'],
                            domain_str_hr       = full_domain_dims_str_hr,
                            crop_region_str_hr  = crop_region_hr_str,
                            hr_scaling_method   = cfg['highres']['scaling_method'],
                            hr_buffer_frac      = cfg['highres']['buffer_frac'] if 'buffer_frac' in cfg['highres'] else 0.0,
                            lr_vars             = cfg['lowres']['condition_variables'],
                            lr_model            = cfg['lowres']['model'],
                            domain_str_lr       = full_domain_dims_str_lr,
                            crop_region_str_lr  = crop_region_lr_str,
                            lr_scaling_methods  = cfg['lowres']['scaling_methods'],
                            lr_buffer_frac      = cfg['lowres']['buffer_frac'] if 'buffer_frac' in cfg['lowres'] else 0.0,
                            split               = scaling_split,
                            stats_dir_root      = cfg['paths']['stats_load_dir'],
                            eps=self.global_prcp_eps
                            )

        logger.info(f"[debug] back-transforms keys: {sorted(list(back_transforms.keys()))}")

        hr_unit, lr_units = get_units(cfg)
        hr_cmap_name, lr_cmap_dict = get_cmaps(cfg)

        p_bar = tqdm.tqdm(gen_dataloader, desc=f"Generating samples for epoch {epoch}", unit="batch")
        try:
            for idx, samples in enumerate(p_bar):
                if 'date' in samples and isinstance(samples['date'], (list, tuple)) and len(samples['date']) > 0:
                    dates = samples['date']
                else:
                    dates = None

                x_gen, y_gen, cond_images_gen, lsm_hr_gen, lsm_gen, sdf_gen, topo_gen, hr_points_gen, lr_points_gen = extract_samples(samples, self.device)

                if idx == 0:
                    try:
                        logger.info("[DEBUG][gen] Batch LR tensor shapes:")
                        for v in self.lr_vars:
                            k_ctx = f"{v}_lr"
                            k_local = f"{v}_lr_local"

                            if k_ctx in samples:
                                logger.info(f"  {k_ctx}: {tuple(samples[k_ctx].shape)}")
                            else:
                                logger.info(f"  {k_ctx}: MISSING")

                            if k_local in samples:
                                logger.info(f"  {k_local}: {tuple(samples[k_local].shape)}")
                            else:
                                logger.info(f"  {k_local}: MISSING")

                    except Exception as e:
                        logger.warning(f"[DEBUG][gen] Failed LR shape print: {e}")

                lsm_cond = lsm_hr_gen if lsm_hr_gen is not None else lsm_gen

                topo_hr_gen = None
                if isinstance(samples, dict) and ("topo_hr" in samples) and (samples["topo_hr"] is not None):
                    topo_hr_gen = samples["topo_hr"].to(self.device)
                topo_cond = topo_hr_gen if topo_hr_gen is not None else topo_gen

                try:
                    local_cond = self._build_local_cond_img(samples)
                except Exception as e:
                    local_cond = cond_images_gen
                    if idx == 0 and epoch == 1:
                        logger.warning(f"[paper2][gen] Failed to build local_cond from batch keys; using cond_images_gen. err={e}")

                try:
                    cond_images_gen = self._build_cond_img(samples)
                except Exception:
                    pass

                if idx == 0 and epoch == 1:
                    try:
                        logger.info(
                            f"[gen][cond] cond_images_gen={None if cond_images_gen is None else tuple(cond_images_gen.shape)}, "
                            f"local_cond={None if local_cond is None else tuple(local_cond.shape)}, "
                            f"lsm_cond={None if lsm_cond is None else tuple(lsm_cond.shape)}, "
                            f"topo_cond={None if topo_cond is None else tuple(topo_cond.shape)}"
                        )
                    except Exception:
                        pass

                logger.info(f"→ Generating {len(x_gen)} samples at epoch {epoch}, batch {idx}...")
                logger.info(f"      Only plotting first {min(cfg['visualization'].get('n_plot_samples', 4), cfg['data_handling']['n_gen_samples'])} samples.")

                lr_ups_baseline = None
                if edm_on and self.edm_predict_residual:
                    lr_ups_baseline = self._build_lr_ups_baseline(local_cond)

                if edm_on and sampler_edm is not None:
                    edm_cfg = cfg.get('edm', {}) or {}
                    guidance_cfg = cfg.get('classifier_free_guidance', {})

                    generated_samples = sampler_edm(score_model=self.model,
                                                batch_size=cfg['data_handling']['n_gen_samples'],
                                                num_steps=edm_cfg.get('sampling_steps', 18),
                                                device=self.device,
                                                img_size=cfg['highres']['data_size'][0],
                                                y=y_gen,
                                                cond_img=cond_images_gen,
                                                lsm_cond=lsm_cond,
                                                topo_cond=topo_cond,
                                                sigma_min=float(edm_cfg.get('sigma_min', 0.002)),
                                                sigma_max=float(edm_cfg.get('sigma_max', 80)),
                                                rho=float(edm_cfg.get('rho', 7.0)),
                                                S_churn=float(edm_cfg.get('S_churn', 0.0)),
                                                S_min=float(edm_cfg.get('S_min', 0.0)),
                                                S_max=float(edm_cfg.get('S_max', float('inf'))),
                                                S_noise=float(edm_cfg.get('S_noise', 1.0)),
                                                lr_ups=lr_ups_baseline,
                                                cfg_guidance=guidance_cfg if guidance_cfg.get('enabled', False) else None,
                                                sigma_star=float(edm_cfg.get('sigma_star', 1.0)),
                    )
                elif sampler is not None:
                    generated_samples = sampler(
                        score_model=self.model,
                        marginal_prob_std=self.marginal_prob_std_fn,
                        diffusion_coeff=self.diffusion_coeff_fn,
                        batch_size=cfg['data_handling']['n_gen_samples'],
                        num_steps=cfg['sampler']['n_timesteps'],
                        device=self.device,
                        img_size=cfg['highres']['data_size'][0],
                        y=y_gen,
                        cond_img=cond_images_gen,
                        lsm_cond=lsm_cond,
                        topo_cond=topo_cond,
                    )
                else:
                    raise ValueError("No valid sampler found. Please check the configuration.")

                gen_model = generated_samples.detach().cpu().float()

                if back_transforms is not None:
                    bt_gen = back_transforms.get(self.bt_gen_key, None)
                    bt_hr = back_transforms.get(self.bt_hr_key, None)
                    if callable(bt_gen):
                        logger.info("[monitor] Applying HR back-transform to generated samples.")
                        gen_phys = bt_gen(gen_model)
                    else:
                        logger.warning("[monitor] HR back-transform not callable; using model space for generated samples.")
                        gen_phys = gen_model

                    if callable(bt_hr):
                        logger.info("[monitor] Applying HR back-transform to ground-truth HR samples.")
                        hr_phys = bt_hr(x_gen)
                    else:
                        logger.warning("[monitor] HR back-transform not callable; using model space for ground-truth HR samples.")
                        hr_phys = x_gen
                else:
                    logger.info("[monitor] No back-transforms available; using model space for generated samples.")
                    gen_phys = gen_model
                    hr_phys = x_gen

                if gen_phys is not None and hr_phys is not None:
                    if not isinstance(gen_phys, torch.Tensor):
                        gen_phys = torch.tensor(gen_phys)
                    tensor_stats(gen_phys, f"eval/x_phys")
                    warn_hi = float(cfg.get('diagnostics', {}).get('warn_if_phys_gt', 300.0))
                    if float(gen_phys.max().item()) > warn_hi:
                        logger.warning(f"[diagnostics][eval] Generated samples exceed {warn_hi} {hr_unit} in physical space. Max: {float(gen_phys.max().item()):.2f} {hr_unit}. Consider adjusting back-transform, data scaling, or adding clamping.")

                try:
                    mon_cfg = cfg.get('monitoring', {}).get('extreme_prcp', {})
                    thr = float(mon_cfg.get('threshold_mm', self.extreme_threshold_mm))

                    if not isinstance(gen_phys, torch.Tensor):
                        gen_phys = torch.tensor(gen_phys)
                    chk = report_precip_extremes(x_bt=gen_phys, name="generated_hr", cap_mm_day=thr)
                    if chk.get('has_extreme', False):
                        extreme_values = chk.get('extreme_values', [])
                        mx = max(extreme_values) if isinstance(extreme_values, list) and extreme_values else None
                        cnt = len(extreme_values) if isinstance(extreme_values, list) else None
                        logger.warning(f"[monitor][gen] Extreme precip: max={mx:.1f} mm/day, count={cnt}, thr={thr} mm/day")

                        if mon_cfg.get('clamp_in_generation', self.extreme_clamp_in_gen):
                            clamp_max = float(mon_cfg.get('clamp_max_mm', thr))
                            gen_phys = torch.clamp(gen_phys, min=0.0, max=clamp_max)
                            logger.warning(f"[monitor][gen] Clamped generated samples to ≤ {clamp_max} mm/day.")
                            logger.warning(f"[monitor][gen] Note: clamping is not done on plotted samples, only on gen_phys used for metrics. Consider adding clamping in sampling instead if desired.")
                except Exception as e:
                    logger.warning(f"[monitor] Could not run extreme sentinel. Error: {e}")

                try:
                    if not isinstance(gen_phys, torch.Tensor):
                        gen_phys = torch.tensor(gen_phys)
                    gen_phys = gen_phys.detach().cpu()
                    if not isinstance(hr_phys, torch.Tensor):
                        hr_phys = torch.tensor(hr_phys)
                    hr_phys = hr_phys.detach().cpu()
                    if lsm_gen is not None:
                        if not isinstance(lsm_gen, torch.Tensor):
                            lsm_gen = torch.tensor(lsm_gen)
                        lsm_gen = lsm_gen.detach().cpu()

                    mask = None
                    if self.eval_land_only and (lsm_gen is not None):
                        mask = (lsm_gen >= 0.5).to(dtype=torch.float32).detach().cpu()

                    if not isinstance(gen_phys, torch.Tensor):
                        gen_phys = torch.tensor(gen_phys)
                    if not isinstance(hr_phys, torch.Tensor):
                        hr_phys = torch.tensor(hr_phys)

                    fss_dict = compute_fss_at_scales(
                        gen_phys, hr_phys, mask=mask,
                        fss_km=self.fss_scales_km,
                        grid_km_per_px=self.pixel_km,
                        thr_mm=self.fss_threshold_mm
                    )
                    self.fss_hist.append(fss_dict)

                    self.epoch_list.append(epoch)

                    plot_fss_history(self.fss_hist, epoch_list=self.epoch_list,
                                    save_dir=self.path_metrics,
                                    filename="fss_history.png",
                                    title="FSS history" + (" (land-only)" if self.eval_land_only else ""),
                                    n_samples=len(gen_phys))
                    psd_dict = compute_psd_slope(gen_phys, hr_bt=hr_phys if self.psd_compare_to_hr else None, mask=mask)
                    self.psd_hist.append(psd_dict)
                    plot_psd_slope_history(self.psd_hist,
                                        epoch_list=self.epoch_list,
                                        save_dir=self.path_metrics,
                                        filename="psd_history.png",
                                        title="PSD slope history" + (" (land-only)" if self.eval_land_only else ""),
                                        n_samples=len(gen_phys))

                    q_dict = compute_p95_p99_and_wet_day(gen_phys,
                                                        hr_bt=hr_phys if self.quantiles_compare_to_hr else None,
                                                        mask=mask,
                                                        wet_threshold_mm=self.wetday_thresh)
                    self.q_hist.append(q_dict)
                    plot_quantiles_wetday_history(self.q_hist, epoch_list=self.epoch_list,
                                                save_dir=self.path_metrics,
                                                filename="quantiles_history.png",
                                                title="Quantiles & wet-day history" + (" (land-only)" if self.eval_land_only else ""),
                                                n_samples=len(gen_phys))
                except Exception as e:
                    logger.warning(f"[monitor] Could not compute epoch-level metrics at epoch {epoch}. Error: {e}")

                if cfg['visualization']['create_figs']:
                    fig, _ = plot_samples_and_generated(
                        samples=samples,
                        generated=gen_model,
                        cfg=cfg,
                        transform_back_bf_plot=cfg['visualization']['transform_back_bf_plot'],
                        back_transforms=back_transforms,
                        dates=dates,
                    )
                    if cfg['visualization']['save_figs']:
                        fig.savefig(os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png'),
                                    dpi=300, bbox_inches='tight')
                        logger.info(f"→ Figure saved to {os.path.join(self.path_figures, f'epoch_{epoch}_generatedSamples.png')}")
                    plt.close(fig)
                    break
        finally:
            model_ref.load_state_dict(model_sd_backup)
            model_ref.train()

    def plot_losses(self,
                    train_losses,
                    val_losses=None,
                    save_path=None,
                    save_name='losses_plot.png',
                    show_plot=False,
                    verbose=True):
        '''
            Plot the training and validation losses.
            Args:
                train_losses: List of training losses.
                val_losses: List of validation losses.
                save_path: Path to save the plot.
                save_name: Name of the plot file.
                show_plot: Boolean to show the plot.
        '''
        # Plot the losses
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses, label='Training Loss', color='blue')
        if val_losses is not None:
            ax.plot(val_losses, label='Validation Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Losses')
        ax.legend()

        # Show the plot
        if show_plot:
            plt.show()
            
        # Save the plot
        if save_path is not None:
            fig.savefig(os.path.join(save_path, save_name), dpi=300, bbox_inches='tight')
            if verbose:
                logger.info(f"→ Losses plot saved to {os.path.join(save_path, save_name)}")

        plt.close(fig)

    def _plot_live_metrics(self, save_dir: str, n_samples: Optional[int] = None):
        """
        Internal method to plot live training metrics if enabled in the configuration.
        Args:
            save_dir (str): Directory where the metrics plot will be saved.
            n_samples (Optional[int]): Number of samples used for computing metrics, for annotation.
        """
        if len(self.live_metrics['steps']) == 0:
            return

        out = os.path.join(self.path_metrics, 'inLoop_metrics_timeseries.png')

        try:
            plot_live_training_metrics(
                self.live_metrics['steps'],
                self.live_metrics['edm_cosine'],
                self.live_metrics['hr_lr_corr'],
                save_dir=self.path_metrics,
                filename='inLoop_metrics_timeseries.png',
                show=self.cfg['visualization'].get('show_figs', False),
                title="In-loop training metrics (EDM cosine, HR-LR corr)",
                land_only=self.eval_land_only,
                n_samples=n_samples
            )
            logger.info(f"→ Live metrics plot saved to {out}")

        except Exception as e:
            logger.error(f"[Monitor] Could not save live metrics plot to {out}. Error: {e}")

        