#!/usr/bin/env python3
"""
test_dataset_transforms.py

Sanity checks for:
- DANRA_Dataset_cutouts_ERA5_Zarr __getitem__
- forward transforms from stats (inside Dataset)
- back transforms from stats (special_transforms.build_back_transforms_from_stats)
- remap_between_scalings / lr_baseline_to_hr_zspace
- dual_lr channel semantics + potential mismatch causing low sums

Run:
  python scripts/test_dataset_transforms.py --cfg path/to/config.yaml --n 5 --idx 0 10 100
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import torch

# If your project uses yaml
import yaml

# Import your dataset + transform helpers
from sbgm.data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from sbgm.special_transforms import (
    build_back_transforms_from_stats,
    load_global_stats,
    lr_baseline_to_hr_zspace,
    get_transforms_from_stats,
    get_backtransforms_from_stats,
)

# ------------------------- helpers -------------------------

def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return None

def _stats(x: Any) -> str:
    a = _to_numpy(x)
    if a is None:
        return "None"
    a = np.asarray(a)
    if a.size == 0:
        return f"empty shape={a.shape}"
    return f"shape={a.shape} dtype={a.dtype} min={np.nanmin(a):.4g} mean={np.nanmean(a):.4g} max={np.nanmax(a):.4g}"

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.nanmean((a - b) ** 2))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.nanmean(np.abs(a - b)))

def _rel_sum_err(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.nansum(a))
    sb = float(np.nansum(b))
    denom = max(abs(sa), 1e-12)
    return float((sb - sa) / denom)

def _print_header(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- main checks -------------------------

@torch.no_grad()
def test_one_sample(
    sample: Dict[str, Any],
    *,
    cfg: dict,
    back_transforms: Dict[str, Any],
    verbose: bool = True,
) -> None:
    hr_var = cfg["highres"]["variable"]
    lr_vars = cfg["lowres"]["condition_variables"]

    if verbose:
        _print_header(f"SAMPLE date={sample.get('date')} idx_points hr={sample.get('hr_points')} lr={sample.get('lr_points')}")
        print(f"Keys: {sorted(sample.keys())}")

    # --- basic prints
    hr_key = f"{hr_var}_hr"
    print(f"\n[HR scaled] {hr_key}: {_stats(sample.get(hr_key))}")
    if f"{hr_var}_hr_original" in sample:
        print(f"[HR orig ] {hr_var}_hr_original: {_stats(sample.get(f'{hr_var}_hr_original'))}")

    for v in lr_vars:
        k = f"{v}_lr"
        print(f"\n[LR scaled] {k}: {_stats(sample.get(k))}")
        if f"{v}_lr_original" in sample:
            print(f"[LR orig ] {v}_lr_original: {_stats(sample.get(f'{v}_lr_original'))}")

    for g in ["lsm", "topo", "lsm_hr", "sdf"]:
        if g in sample:
            print(f"\n[GEO] {g}: {_stats(sample.get(g))}")

    # --- round-trip tests using back_transforms dict
    # HR round-trip: orig -> fwd(hr stats) -> inv(hr stats)  (requires orig)
    if f"{hr_var}_hr_original" in sample and sample.get(f"{hr_var}_hr_original") is not None:
        hr_orig = sample[f"{hr_var}_hr_original"]
        if torch.is_tensor(hr_orig):
            hr_orig_t = hr_orig.to(torch.float32)
        else:
            hr_orig_t = torch.tensor(hr_orig, dtype=torch.float32)

        # forward transform used in Dataset is not directly accessible here;
        # rebuild forward from stats (same params as dataset)
        domain_str_hr = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg["highres"]["full_domain_dims"] is not None else "full_domain"
        crop_hr = cfg["highres"]["cutout_domains"]
        crop_region_str_hr = "_".join(map(str, crop_hr)) if crop_hr is not None else "full"
        split = cfg["transforms"].get("scaling_split", "train")
        stats_root = cfg["paths"]["stats_load_dir"]
        hr_buffer = float(cfg["highres"].get("buffer_frac", 0.0))
        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if hr_var in ["prcp", "tp", "cape"] else 0.0

        fwd_hr = get_transforms_from_stats(
            variable=hr_var,
            model=cfg["highres"]["model"],
            domain_str=domain_str_hr,
            crop_region_str=crop_region_str_hr,
            scaling_split=split,
            transform_type=cfg["highres"]["scaling_method"],
            buffer_frac=hr_buffer,
            stats_file_path=stats_root,
            eps=eps,
        )
        inv_hr = back_transforms[hr_key]

        hr_norm = fwd_hr(hr_orig_t)
        hr_rec = inv_hr(hr_norm)

        a = _to_numpy(hr_orig_t)
        b = _to_numpy(hr_rec)
        print("\n[ROUNDTRIP HR] orig -> fwd(HR) -> inv(HR)")
        print(f"  MSE={_mse(a,b):.6g}  MAE={_mae(a,b):.6g}  rel_sum_err={(100*_rel_sum_err(a,b)):.3f}%") # type: ignore

    # LR round-trip tests for each condition if original exists
    domain_str_lr = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg["lowres"]["full_domain_dims"] is not None else "full_domain"
    crop_lr = cfg["lowres"]["cutout_domains"]
    crop_region_str_lr = "_".join(map(str, crop_lr)) if crop_lr is not None else "full"
    split = cfg["transforms"].get("scaling_split", "train")
    stats_root = cfg["paths"]["stats_load_dir"]
    lr_buffer = float(cfg["lowres"].get("buffer_frac", 0.0))

    for v, mth in zip(lr_vars, cfg["lowres"]["scaling_methods"]):
        if f"{v}_lr_original" not in sample or sample.get(f"{v}_lr_original") is None:
            continue

        lr_orig = sample[f"{v}_lr_original"]
        lr_orig_t = torch.tensor(lr_orig, dtype=torch.float32) if not torch.is_tensor(lr_orig) else lr_orig.to(torch.float32)

        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if v in ["prcp", "tp", "cape"] else 0.0
        fwd_lr = get_transforms_from_stats(
            variable=v,
            model=cfg["lowres"]["model"],
            domain_str=domain_str_lr,
            crop_region_str=crop_region_str_lr,
            scaling_split=split,
            transform_type=mth,
            buffer_frac=lr_buffer,
            stats_file_path=stats_root,
            eps=eps,
        )
        inv_lr = back_transforms[f"{v}_lr"]

        lr_norm = fwd_lr(lr_orig_t)
        lr_rec = inv_lr(lr_norm)

        a = _to_numpy(lr_orig_t)
        b = _to_numpy(lr_rec)
        print(f"\n[ROUNDTRIP LR] {v}: orig -> fwd(LR) -> inv(LR)")
        print(f"  MSE={_mse(a,b):.6g}  MAE={_mae(a,b):.6g}  rel_sum_err={(100*_rel_sum_err(a,b)):.3f}%") # type: ignore

    # --- Dual-LR mismatch probe (the exact failure mode you suspect)
    # If main variable exists in LR and sample[v_lr] has 2 channels: compare inverse behavior.
    main = cfg["highres"]["variable"]
    if main in lr_vars and torch.is_tensor(sample.get(f"{main}_lr")):
        lr_scaled = sample[f"{main}_lr"]
        if lr_scaled.ndim == 3 and lr_scaled.shape[0] == 2:
            print("\n[DUAL-LR] main LR has 2 channels. Testing inversion consequences.")
            ch0 = lr_scaled[0]
            ch1 = lr_scaled[1]

            # Invert both with the LR backtransform (this is what plotting/eval often does)
            inv_lr_main = back_transforms[f"{main}_lr"]
            x0 = inv_lr_main(ch0)
            x1 = inv_lr_main(ch1)

            print(f"  inv(LR) on channel0: {_stats(x0)}  sum={float(torch.sum(x0)):.4g}")
            print(f"  inv(LR) on channel1: {_stats(x1)}  sum={float(torch.sum(x1)):.4g}")

            # Now: take HR-scaled version (if available) and invert with LR inv — should show the “low sum / clamp” effect
            if torch.is_tensor(sample.get(f"{main}_hr")):
                hr_scaled = sample[f"{main}_hr"]
                x_bad = inv_lr_main(hr_scaled.squeeze(0) if hr_scaled.ndim == 3 and hr_scaled.shape[0] == 1 else hr_scaled)
                print("\n  [MISMATCH TEST] inv(LR) applied to HR-normalized field")
                print(f"    {_stats(x_bad)}  sum={float(torch.sum(x_bad)):.4g}")

    # --- Baseline remap test (LR norm -> HR z-space) if shapes allow
    # This is the “right” mapping for EDM residual baseline.
    if main in lr_vars and torch.is_tensor(sample.get(f"{main}_lr")) and torch.is_tensor(sample.get(f"{main}_hr")):
        # choose channel 1 if dual-lr, else single
        lr_scaled = sample[f"{main}_lr"]
        lr_chan = lr_scaled[1] if (lr_scaled.ndim == 3 and lr_scaled.shape[0] == 2) else (lr_scaled.squeeze(0) if lr_scaled.ndim == 3 else lr_scaled)

        hr_var = cfg["highres"]["variable"]
        hr_model = cfg["highres"]["model"]
        lr_model = cfg["lowres"]["model"]

        hr_domain_str = f"{cfg['highres']['full_domain_dims'][0]}x{cfg['highres']['full_domain_dims'][1]}" if cfg["highres"]["full_domain_dims"] is not None else "full_domain"
        lr_domain_str = f"{cfg['lowres']['full_domain_dims'][0]}x{cfg['lowres']['full_domain_dims'][1]}" if cfg["lowres"]["full_domain_dims"] is not None else "full_domain"
        hr_crop = cfg["highres"]["cutout_domains"]
        lr_crop = cfg["lowres"]["cutout_domains"]
        hr_crop_str = "_".join(map(str, hr_crop)) if hr_crop is not None else "full"
        lr_crop_str = "_".join(map(str, lr_crop)) if lr_crop is not None else "full"
        split = cfg["transforms"].get("scaling_split", "train")
        stats_root = cfg["paths"]["stats_load_dir"]
        hr_buffer = float(cfg["highres"].get("buffer_frac", 0.0))
        lr_buffer = float(cfg["lowres"].get("buffer_frac", 0.0))
        lr_method = cfg["lowres"]["scaling_methods"][lr_vars.index(main)]
        hr_method = cfg["highres"]["scaling_method"]
        eps = float(cfg["transforms"].get("prcp_eps", 0.01)) if main in ["prcp", "tp", "cape"] else 0.0

        mapped = lr_baseline_to_hr_zspace(
            lr_chan_norm=lr_chan,
            lr_variable=main,
            lr_model=lr_model,
            lr_domain_str=lr_domain_str,
            lr_crop_region_str=lr_crop_str,
            lr_split=split,
            lr_scaling_method=lr_method,
            lr_buffer_frac=lr_buffer,
            lr_stats_dir_root=stats_root,
            hr_variable=hr_var,
            hr_model=hr_model,
            hr_domain_str=hr_domain_str,
            hr_crop_region_str=hr_crop_str,
            hr_split=split,
            hr_buffer_frac=hr_buffer,
            hr_stats_dir_root=stats_root,
            hr_scaling_method=hr_method,
            eps=eps,
        )

        print("\n[REMAP] lr_baseline_to_hr_zspace output stats (should look like HR-normalized distribution):")
        print(f"  mapped: {_stats(mapped)}")
        print(f"  hr_scaled: {_stats(sample[f'{main}_hr'])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="Path to your YAML config.")
    ap.add_argument("--n", type=int, default=3, help="Number of samples to test (if --idx not given).")
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="Explicit indices to test.")
    ap.add_argument("--no_cutouts", action="store_true", help="Force cutouts=False to simplify debugging.")
    ap.add_argument("--save_original", action="store_true", help="Force dataset save_original=True to enable roundtrip tests.")
    ap.add_argument("--verbose", action="store_true", help="Extra prints.")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Prepare minimal geo inputs if your dataset expects them
    # If your cfg points to lsm/topo paths elsewhere, you can load them here.
    # For now we attempt to load from cfg paths if present.
    lsm = None
    topo = None
    if "paths" in cfg:
        lsm_path = cfg["paths"].get("lsm_path", None)
        topo_path = cfg["paths"].get("topo_path", None)
        if lsm_path and os.path.exists(lsm_path):
            lsm = np.flipud(np.load(lsm_path)["data"]).copy()
        if topo_path and os.path.exists(topo_path):
            topo = np.flipud(np.load(topo_path)["data"]).copy()

    # ---- Build dataset paths (you likely already have these in your pipeline;
    # Here we assume cfg['paths']['data_dir'] etc. is consistent with your build_data_path helper.
    # If you prefer, pass direct zarr paths here.
    #
    # IMPORTANT: replace these with your own path builder if needed.

    full_domain_dims_hr = cfg["highres"].get("full_domain_dims", None)
    full_domain_dims_lr = cfg["lowres"].get("full_domain_dims", None)

    from sbgm.utils import build_data_path
    hr_zarr = build_data_path(cfg['paths']['data_dir'], cfg['highres']['model'], cfg['highres']['variable'], full_domain_dims_hr, 'train')
    lr_zarr_dict = {}
    for v in cfg["lowres"]["condition_variables"]:
        lr_zarr = build_data_path(cfg['paths']['data_dir'], cfg['lowres']['model'], v, full_domain_dims_lr, 'train')
        lr_zarr_dict[v] = lr_zarr
    # hr_zarr = cfg["paths"]["hr_zarr_dir"] if "hr_zarr_dir" in cfg["paths"] else None
    # lr_zarr_dict = cfg["paths"].get("lr_zarr_dirs", None)  # expect dict {var: path}

    if hr_zarr is None or lr_zarr_dict is None:
        raise ValueError(
            "This script expects cfg['paths']['hr_zarr_dir'] and cfg['paths']['lr_zarr_dirs'] (dict). "
            "Either add them to your config or modify the script to use your build_data_path() logic."
        )

    # ---- Instantiate dataset
    dcfg = cfg.copy()
    if args.no_cutouts:
        dcfg["transforms"]["sample_w_cutouts"] = False
    if args.save_original:
        dcfg["visualization"]["show_both_orig_scaled"] = True  # matches your save_original usage

    ds = DANRA_Dataset_cutouts_ERA5_Zarr(
        hr_variable_dir_zarr=hr_zarr,
        hr_data_size=tuple(cfg["highres"]["data_size"]) if cfg["highres"]["data_size"] is not None else (128, 128),
        n_samples=int(cfg.get("data_handling", {}).get("n_samples_debug", 200)),
        cache_size=int(cfg.get("data_handling", {}).get("cache_size", 0)),
        hr_variable=cfg["highres"]["variable"],
        hr_model=cfg["highres"]["model"],
        hr_scaling_method=cfg["highres"]["scaling_method"],
        lr_conditions=cfg["lowres"]["condition_variables"],
        lr_model=cfg["lowres"]["model"],
        lr_scaling_methods=cfg["lowres"]["scaling_methods"],
        lr_cond_dirs_zarr=lr_zarr_dict,
        geo_variables=cfg.get("stationary_conditions", {}).get("geographic_conditions", {}).get("geo_variables", ["lsm", "topo"]),
        lsm_full_domain=lsm,
        topo_full_domain=topo,
        conditional_seasons=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("sample_w_cond_season", False)),
        use_sin_cos_embedding=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("use_sin_cos_embedding", False)),
        use_leap_years=bool(cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("use_leap_years", True)),
        cfg=dcfg,
        split=cfg.get("transforms", {}).get("scaling_split", "train"),
        shuffle=False,
        cutouts=bool(cfg.get("transforms", {}).get("sample_w_cutouts", False)),
        cutout_domains=cfg["highres"].get("cutout_domains", None),
        sdf_weighted_loss=bool(cfg.get("stationary_conditions", {}).get("geographic_conditions", {}).get("sample_w_sdf", False)),
        scale=bool(cfg.get("transforms", {}).get("scaling", True)),
        save_original=bool(dcfg.get("visualization", {}).get("show_both_orig_scaled", False)),
        n_classes=cfg.get("stationary_conditions", {}).get("seasonal_conditions", {}).get("n_seasons", None),
        lr_data_size=tuple(cfg["lowres"]["data_size"]) if cfg["lowres"]["data_size"] is not None else None,
        lr_cutout_domains=cfg["lowres"].get("cutout_domains", None),
        resize_factor=int(cfg["lowres"].get("resize_factor", 1)),
        fixed_cutout_hr=bool(cfg["highres"].get("stationary_cutout", {}).get("enabled", False)),
        fixed_hr_bounds=cfg["highres"].get("stationary_cutout", {}).get("bounds", None),
        fixed_cutout_lr=bool(cfg["lowres"].get("stationary_cutout", {}).get("enabled", False)),
        fixed_lr_bounds=cfg["lowres"].get("stationary_cutout", {}).get("bounds", None),
    )

    print(f"Dataset length: {len(ds)}")

    # ---- Build back-transforms using the same helper as training.py
    full_domain_dims_hr = cfg["highres"].get("full_domain_dims", None)
    full_domain_dims_lr = cfg["lowres"].get("full_domain_dims", None)

    domain_str_hr = f"{full_domain_dims_hr[0]}x{full_domain_dims_hr[1]}" if full_domain_dims_hr is not None else "full_domain"
    domain_str_lr = f"{full_domain_dims_lr[0]}x{full_domain_dims_lr[1]}" if full_domain_dims_lr is not None else "full_domain"

    crop_region_hr = cfg["highres"].get("cutout_domains", None)
    crop_region_lr = cfg["lowres"].get("cutout_domains", None)
    crop_region_str_hr = "_".join(map(str, crop_region_hr)) if crop_region_hr is not None else "full"
    crop_region_str_lr = "_".join(map(str, crop_region_lr)) if crop_region_lr is not None else "full"

    back_transforms = build_back_transforms_from_stats(
        hr_var=cfg["highres"]["variable"],
        hr_model=cfg["highres"]["model"],
        domain_str_hr=domain_str_hr,
        crop_region_str_hr=crop_region_str_hr,
        hr_scaling_method=cfg["highres"]["scaling_method"],
        hr_buffer_frac=float(cfg["highres"].get("buffer_frac", 0.0)),
        lr_vars=cfg["lowres"]["condition_variables"],
        lr_model=cfg["lowres"]["model"],
        lr_scaling_methods=cfg["lowres"]["scaling_methods"],
        domain_str_lr=domain_str_lr,
        crop_region_str_lr=crop_region_str_lr,
        lr_buffer_frac=float(cfg["lowres"].get("buffer_frac", 0.0)),
        split=cfg["transforms"].get("scaling_split", "train"),
        stats_dir_root=cfg["paths"]["stats_load_dir"],
        eps=float(cfg["transforms"].get("prcp_eps", 0.01)),
    )

    # ---- Extra: print log clamp ranges for prcp/tp/cape if present
    def _print_log_ranges(model: str, var: str):
        s = load_global_stats(
            variable=var,
            model=model,
            domain_str=domain_str_lr if model == cfg["lowres"]["model"] else domain_str_hr,
            crop_region_str=crop_region_str_lr if model == cfg["lowres"]["model"] else crop_region_str_hr,
            split=cfg["transforms"].get("scaling_split", "train"),
            dir_load=cfg["paths"]["stats_load_dir"],
        )
        if not s:
            return
        if "log_min" in s and "log_max" in s:
            print(f"  stats {model}/{var}: log_min={s['log_min']:.4g} log_max={s['log_max']:.4g} log_mean={s.get('log_mean', None)} log_std={s.get('log_std', None)}")

    print("\n[STATS clamp ranges]")
    for v in set(cfg["lowres"]["condition_variables"] + [cfg["highres"]["variable"]]):
        if v in ["prcp", "tp", "cape"]:
            _print_log_ranges(cfg["highres"]["model"], v)
            _print_log_ranges(cfg["lowres"]["model"], v)

    # ---- Select indices
    if args.idx is not None and len(args.idx) > 0:
        indices = args.idx
    else:
        indices = list(range(min(args.n, len(ds))))

    for i in indices:
        sample = ds[i]
        test_one_sample(sample, cfg=cfg, back_transforms=back_transforms, verbose=args.verbose)

if __name__ == "__main__":
    main()