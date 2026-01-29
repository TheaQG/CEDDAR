from __future__ import annotations
from pathlib import Path
import logging
import os
from omegaconf import OmegaConf

from baselines.adapter import BaselineAdapter
from baselines.bilinear import run_bilinear
from baselines.quantile_mapping import run_qm
from baselines.unet_sr.train import run_unet_sr
from baselines.build_dataset import build_dataset_for_split

logger = logging.getLogger(__name__)
logger.info("[Baseline] baseline_main loaded.")
# ---------------------------
# Helpers
# ---------------------------
def _normalize_split(split: str) -> str:
    """Map common aliases to dataset-expected tags."""
    if split.lower() in {"val", "valid"}:
        return "valid"
    if split.lower() in {"test", "gen"}:
        return "test"  # build_dataset_for_split accepts "test" (internally maps to dataset 'gen')
    return "train"

def build_dataset_from_cfg(cfg, split: str, scale: bool = False):
    """
        Build a dataset for the given split by delegating to build_dataset_for_split.
        Adds basic logging of split normalization and dataset size. 
    """
    split_norm = _normalize_split(split)
    logger.info(f"[Baseline] Building dataset for split '{split_norm}', (normalized: '{split_norm}')")
    try:
        ds = build_dataset_for_split(cfg, split_norm, verbose=True, scale=scale)
        try:
            n = len(ds)
        except Exception:
            n = "unknown"
        logger.info(f"[Baseline] Dataset ready: split: '{split_norm}', size: {n} samples")
        return ds
    except Exception as e:
        logger.error(f"Failed to build dataset for split '{split_norm}': {e}")
        raise

def build_adapter(cfg, split: str, extra_channels=None, scale: bool = False):
    """
    Instantiate BaselineAdapter over the dataset for `split`, and auto-pick HR-aligned extras.
    """
    logger.info(f"[Baseline] Building adapter for split '{split}'")
    ds = build_dataset_from_cfg(cfg, split, scale=scale)
    if extra_channels is None:
        extra = []
        probe = ds[0]
        if probe is None:
            logger.warning(f"[Baseline] Probe on dataset[0] returned None; extra channel auto-detection may be incomplete.")
        # prefer HR-aligned geo channels if present
        if 'topo_hr' in probe:
            extra.append('topo_hr')
        elif 'topo' in probe:
            extra.append('topo')
        if 'lsm_hr' in probe:
            extra.append('lsm_hr')
    else:
        extra = list(extra_channels)
    hr_var = cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var','prcp'))
    logger.info(f"[Baseline] Adapter ready: hr_var='{hr_var}', extra_channels={extra}")
    return BaselineAdapter(ds, hr_var=hr_var, extra_channels=extra)

# ---------------------------
# Entrypoint
# ---------------------------
def run(cfg):
    """
    Dispatch entrypoint for baselines.
    cfg.baseline.type in {'bilinear', 'qm', 'unet_sr'}
    Uses build_dataset_for_split under the hood to construct datasets per split.
    """
    # Robust config access with defaults (works for DictConfig or plain dict)
    base = cfg.get('baseline', {})
    # allow env-var overrides for batch jobs
    btype = str(os.getenv('BASELINE_TYPE', base.get('type', 'all'))).lower()

    # support either a single 'split' or a list in 'splits'
    splits_cfg = base.get('splits', None)
    if splits_cfg is None:
        split_single = base.get('split', 'test')
        splits = [split_single]
    else:
        splits = list(splits_cfg)

    logger.info(f"[Baseline] run() called with baseline.type='{btype}', splits={splits}")

    def _run_single(bt: str, split_norm: str):
        bt = bt.lower()
        save_root = Path(cfg.paths.sample_dir) / 'generation' / 'baselines' / bt / split_norm
        save_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Baseline] Save root for '{bt}' baseline: {save_root}")
        if bt == 'bilinear':
            scale = False # Do not scale dataset for bilinear baseline
            adapter = build_adapter(cfg, split_norm, scale=scale)
            logger.info("[Baseline] Launching bilinear baseline...")
            return ('bilinear', run_bilinear(OmegaConf.to_container(cfg, resolve=True), adapter, split_name=split_norm, out_root=save_root))
        elif bt == 'qm':
            scale = False # Do not scale dataset for quantile mapping baseline
            adapter_train = build_adapter(cfg, 'train', scale=scale)
            adapter_apply = build_adapter(cfg, split_norm, scale=scale)
            logger.info("[Baseline] Launching quantile mapping baseline...")
            logger.info(f"[Baseline] QM fit on 'train' split, apply on '{split_norm}' split.")
            return ('qm', run_qm(OmegaConf.to_container(cfg, resolve=True), adapter_train, adapter_apply, split_name=split_norm, out_root=save_root))
        elif bt == 'unet_sr':
            scale = True # Scale dataset for unet_sr baseline
            adapter_tr  = build_adapter(cfg, 'train', scale=scale)
            adapter_val = build_adapter(cfg, 'valid', scale=scale)
            adapter_te  = build_adapter(cfg, 'test', scale=scale)
            logger.info("[Baseline] Launching UNet-SR baseline...")
            logger.info(f"[Baseline]    - Save root: {save_root}")
            return ('unet_sr', run_unet_sr(OmegaConf.to_container(cfg, resolve=True), adapter_tr, adapter_val, adapter_te, save_root))
        else:
            raise ValueError(f"Unknown baseline type: {bt}")

    if btype in {'all', 'auto', 'everything'}:
        results = {}
        order = ['bilinear', 'qm', 'unet_sr']
        for split in splits:
            split_norm = _normalize_split(split)
            logger.info(f"[Baseline] Running ALL baselines on split '{split_norm}'")
            for bt in order:
                logger.info(f"[Baseline] === Starting '{bt}' ({split_norm}) ===")
                key, res = _run_single(bt, split_norm)
                results[(bt, split_norm)] = res
                logger.info(f"[Baseline] === Finished '{bt}' ({split_norm}) ===")
        logger.info("[Baseline] All baselines completed.")
        return results

    result = None
    for split in splits:
        split_norm = _normalize_split(split)
        logger.info(f"[Baseline] Running baseline type '{btype}' on split '{split_norm}'")
        _, result = _run_single(btype, split_norm)

    return result
