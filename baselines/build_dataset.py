import logging
import zarr # type: ignore
import numpy as np
from typing import Optional
from ceddar.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from ceddar.utils import build_data_path
from ceddar.variable_utils import get_units

logger = logging.getLogger(__name__)


def build_dataset_for_split(cfg, split: str, verbose: bool = True, scale: bool = False):
    """
    Construct a DANRA_Dataset_cutouts_ERA5_Zarr for a given split.

    Args:
        cfg (dict or OmegaConf): your project config (same structure used by get_dataloader)
        split (str): one of {"train","valid","test","gen"}
                     ("test" and "gen" are treated identically for paths; "gen" uses evaluation cutout knobs)
        verbose (bool): extra logging

    Returns:
        dataset (DANRA_Dataset_cutouts_ERA5_Zarr)
    """
    assert split in {"train", "valid", "test", "gen"}, f"Unknown split: {split}"

    # ------------------------------- Units/info (purely cosmetic) -------------------------------
    hr_unit, lr_units = get_units(cfg)
    logger.info(f"\nUsing HR data type: {cfg['highres']['model']} {cfg['highres']['variable']} [{hr_unit}]")
    for i, cond in enumerate(cfg['lowres']['condition_variables']):
        logger.info(f"Using LR data type {i+1}: {cfg['lowres']['model']} {cond} [{lr_units[i]}]")

    # ------------------------------- Spatial sizes -------------------------------
    hr_data_size = tuple(cfg['highres']['data_size']) if cfg['highres']['data_size'] is not None else (128, 128)
    lr_data_size = tuple(cfg['lowres']['data_size']) if cfg['lowres']['data_size'] is not None else None
    lr_data_size_use = lr_data_size if lr_data_size is not None else hr_data_size

    if cfg['lowres']['resize_factor'] > 1:
        sf = cfg['lowres']['resize_factor']
        hr_data_size_use = (hr_data_size[0] // sf, hr_data_size[1] // sf)
        lr_data_size_use = (lr_data_size_use[0] // sf, lr_data_size_use[1] // sf)
    else:
        hr_data_size_use = hr_data_size

    if verbose:
        logger.info(f"\nHigh-resolution data size: {hr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tHigh-resolution data size after resize: {hr_data_size_use}")
        logger.info(f"Low-resolution data size: {lr_data_size_use}")
        if cfg['lowres']['resize_factor'] > 1:
            logger.info(f"\tLow-resolution data size after resize: {lr_data_size_use}")

    # ------------------------------- Full-domain dims -------------------------------
    full_domain_dims = tuple(cfg['highres']['full_domain_dims']) if cfg['highres']['full_domain_dims'] is not None else None

    # ------------------------------- Zarr paths per split -------------------------------
    # Helper to map split -> zarr group path
    def _hr_dir_for_split(s: str) -> str:
        tag = "train" if s == "train" else ("valid" if s == "valid" else "test")
        return build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'],
                               cfg['highres']['variable'], full_domain_dims, tag)

    def _lr_dirs_for_split(s: str) -> dict:
        tag = "train" if s == "train" else ("valid" if s == "valid" else "test")
        d = {}
        for cond in cfg['lowres']['condition_variables']:
            d[cond] = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'],
                                      cond, full_domain_dims, tag)
        return d

    hr_data_dir = _hr_dir_for_split(split)
    lr_cond_dirs = _lr_dirs_for_split(split)

    # ------------------------------- Geo (lsm/topo) handling -------------------------------
    if cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf']:
        logger.info('SDF weighted loss enabled. Setting lsm and topo to true.\n')
        sample_w_geo = True
    else:
        sample_w_geo = cfg['stationary_conditions']['geographic_conditions']['sample_w_geo']

    if sample_w_geo:
        logger.info('Using geographical features for sampling.\n')
        geo_variables = cfg['stationary_conditions']['geographic_conditions']['geo_variables']
        data_dir_lsm = cfg['paths']['lsm_path']
        data_dir_topo = cfg['paths']['topo_path']

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])

        if cfg['transforms']['scaling']:
            # optional normalization of topo to the same lsm range
            topo_min = cfg['stationary_conditions']['geographic_conditions']['topo_min']
            topo_max = cfg['stationary_conditions']['geographic_conditions']['topo_max']
            norm_min = cfg['stationary_conditions']['geographic_conditions']['norm_min']
            norm_max = cfg['stationary_conditions']['geographic_conditions']['norm_max']

            if topo_min is None or topo_max is None:
                topo_min, topo_max = np.min(data_topo), np.max(data_topo)
            if norm_min is None or norm_max is None:
                norm_min, norm_max = np.min(data_lsm), np.max(data_lsm)

            OldRange = (topo_max - topo_min) if (topo_max - topo_min) != 0 else 1.0
            NewRange = (norm_max - norm_min)
            data_topo = ((data_topo - topo_min) * NewRange / OldRange) + norm_min
    else:
        geo_variables = None
        data_lsm = None
        data_topo = None

    # ------------------------------- Cutouts & stationary cutouts -------------------------------
    # Default DK-ish crop if unspecified
    cutout_domains = tuple(cfg['highres']['cutout_domains']) if cfg['highres']['cutout_domains'] is not None else (170, 350, 340, 520)
    lr_cutout_domains = tuple(cfg['lowres']['cutout_domains']) if cfg['lowres']['cutout_domains'] is not None else (170, 350, 340, 520)

    # Training/validation stationary knobs
    stationary_cutout_hr = bool(cfg['highres'].get('stationary_cutout', {}).get('enabled', False))
    hr_bounds = cfg['highres'].get('stationary_cutout', {}).get('hr_bounds', None)
    stationary_cutout_lr = bool(cfg['lowres'].get('stationary_cutout', {}).get('enabled', False))
    lr_bounds = cfg['lowres'].get('stationary_cutout', {}).get('lr_bounds', None)

    # Generation/evaluation stationary knobs (used for split == "gen" or "test")
    stationary_cutout_gen_hr = bool(cfg['evaluation'].get('stationary_cutout', {}).get('hr_enabled', False))
    hr_bounds_gen = cfg['evaluation'].get('stationary_cutout', {}).get('hr_bounds', None)
    stationary_cutout_gen_lr = bool(cfg['evaluation'].get('stationary_cutout', {}).get('lr_enabled', False))
    lr_bounds_gen = cfg['evaluation'].get('stationary_cutout', {}).get('lr_bounds', None)

    # Which stationary knobs to use for this split
    if split in {"test", "gen"}:
        fixed_cutout_hr = stationary_cutout_gen_hr
        fixed_hr_bounds = hr_bounds_gen
        fixed_cutout_lr = stationary_cutout_gen_lr
        fixed_lr_bounds = lr_bounds_gen
        shuffle = False
    else:
        fixed_cutout_hr = stationary_cutout_hr
        fixed_hr_bounds = hr_bounds
        fixed_cutout_lr = stationary_cutout_lr
        fixed_lr_bounds = lr_bounds
        shuffle = True if split == "train" else True  # keep val shuffling if you like; set False if you prefer deterministic

    # ------------------------------- Seasonal classification -------------------------------
    if cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season']:
        n_seasons = cfg['stationary_conditions']['seasonal_conditions']['n_seasons']
    else:
        n_seasons = None

    # ------------------------------- Zarr open and sample counts -------------------------------
    data_zarr = zarr.open_group(hr_data_dir, mode='r')
    n_samples = len(list(data_zarr.keys()))

    # Cache sizing
    if cfg.get('data_handling', {}).get('cache_size', 0) == 0:
        cache_size = n_samples // 2
    else:
        cache_size = cfg.get('data_handling', {}).get('cache_size', 0)

    if verbose:
        logger.info(f"\n[{split}] n_samples={n_samples}, cache_size={cache_size}")

    # ------------------------------- Dataset construction -------------------------------
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(
        hr_variable_dir_zarr = hr_data_dir,
        hr_data_size         = hr_data_size_use,
        n_samples            = n_samples,
        cache_size           = cache_size,
        hr_variable          = cfg['highres']['variable'],
        hr_model             = cfg['highres']['model'],
        hr_scaling_method    = cfg['highres']['scaling_method'],
        # hr_scaling_params  = cfg['highres']['scaling_params'],  # if you use external stats
        lr_conditions        = cfg['lowres']['condition_variables'],
        lr_model             = cfg['lowres']['model'],
        lr_scaling_methods   = cfg['lowres']['scaling_methods'],
        # lr_scaling_params  = cfg['lowres']['scaling_params'],
        lr_cond_dirs_zarr    = lr_cond_dirs,
        geo_variables        = geo_variables,
        lsm_full_domain      = data_lsm,
        topo_full_domain     = data_topo,
        cfg                  = cfg,
        split                = ("gen" if split == "test" else split),  # your dataset uses "gen" for test/eval
        shuffle              = shuffle,
        cutouts              = cfg['transforms']['sample_w_cutouts'],
        cutout_domains       = list(cutout_domains) if cfg['transforms']['sample_w_cutouts'] else None,
        n_samples_w_cutouts  = n_samples,
        sdf_weighted_loss    = cfg['stationary_conditions']['geographic_conditions']['sample_w_sdf'],
        scale                = scale,
        save_original        = cfg['visualization']['show_both_orig_scaled'],
        conditional_seasons  = cfg['stationary_conditions']['seasonal_conditions']['sample_w_cond_season'],
        n_classes            = n_seasons,
        lr_data_size         = tuple(lr_data_size_use) if lr_data_size_use is not None else None,
        lr_cutout_domains    = list(lr_cutout_domains) if lr_cutout_domains is not None else None,
        resize_factor        = cfg['lowres']['resize_factor'],
        fixed_cutout_hr      = bool(fixed_cutout_hr),
        fixed_hr_bounds      = fixed_hr_bounds,
        fixed_cutout_lr      = bool(fixed_cutout_lr),
        fixed_lr_bounds      = fixed_lr_bounds,
    )

    return dataset