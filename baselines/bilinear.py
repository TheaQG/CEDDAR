# sbgm/baselines/bilinear.py

from __future__ import annotations # for Python 3.7 compatibility
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional
import torch

from baselines.plotting import plotting_enabled, plotting_params, resolve_samples_dir, plot_triplet

logger = logging.getLogger(__name__)

def _save_npz_compressed(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: np.array(v) for k, v in arrays.items() if v is not None}
    np.savez_compressed(path, **out)

def run_bilinear(cfg, adapter, out_root, split_name: str = "test", quicklook: bool = False):
    """
        Uses lr_up directly as the baseline prediction and saves to a generation-like folder.
        Folder structure mirrors generation.py:
            out_root/
                pmm/{date}.npz -> {'pmm': [1,1,H,W]}
                lr_hr/{date}.npz -> {'hr': [1,1,H,W] or None, 'lr': [1,1,H,W] or None}
                lsm/{date}.npz -> {'lsm': [1,1,H,W]} [if available]
                meta/manifest.json
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.get('baseline', {}).get('experiment_name', 'bilinear_baseline')
    hr_var = cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var','prcp'))

    logger.info(f"[Bilinear] Starting run: name='{run_name}', split='{split_name}'")
    logger.info(f"[Bilinear] Output root: {out_root}")

    if not quicklook:
        for d in ['pmm', 'lr_hr', 'lsm', 'meta']:
            (out_root / d).mkdir(parents=True, exist_ok=True)

    loader = adapter.make_loader(batch_size=cfg.get('batch_size', 16), shuffle=False, num_workers=cfg.get('num_workers', 4))

    # plotting setup (optional)
    want_plot = plotting_enabled(cfg)
    params = plotting_params(cfg) if want_plot else {}
    plotted = 0
    if want_plot:
        sample_dir = resolve_samples_dir(cfg, baseline_type="bilinear", split=split_name)
        logger.info(f"[Bilinear][plot] Enabled. Saving examples to: {sample_dir}")
    else:
        sample_dir = None

    n_days = 0
    results = []
    for b_idx, batch in enumerate(loader):
        dates, lr_up, y, lsm = batch.date, batch.lr_up, batch.y, batch.lsm
        if b_idx == 0:
            B, _, H, W = lr_up.shape
            logger.info(f"[Bilinear] First batch shape: B={B}, HxW={H}x{W}")
        if (b_idx + 1) % 20 == 0:
            logger.info(f"[Bilinear] Processing batch {b_idx+1}")

        lr_np = lr_up.cpu().numpy()  # (B,1,H,W)
        y_np = y.cpu().numpy() if y is not None else None
        lsm_np = lsm.cpu().numpy() if lsm is not None else None

        for i in range(lr_np.shape[0]):
            date0 = dates[i]
            pmm = lr_np[i:i+1]  # (1,1,H,W)
            hr = y_np[i:i+1] if y_np is not None else None
            lr = lr_np[i:i+1]  # (1,1,H,W)
            lsm_i = lsm_np[i:i+1] if lsm_np is not None else None

            if quicklook:
                results.append({"date": date0, "pmm": pmm, "hr": hr, "lr": lr})
            else:
                _save_npz_compressed(out_root / 'pmm' / f"{date0}.npz", pmm=pmm)
                _save_npz_compressed(out_root / 'lr_hr' / f"{date0}.npz", hr=hr, lr=lr)
                if lsm_i is not None:
                    _save_npz_compressed(out_root / 'lsm' / f"{date0}.npz", lsm=lsm_i)
            n_days += 1

            # optional plotting
            max_plots = int(params.get("max_plots", 0))
            if want_plot and plotted < max_plots and sample_dir is not None:
                try:
                    vmax_param = params.get("vmax", None)
                    if isinstance(vmax_param, str):
                        try:
                            vmax_param = float(vmax_param)
                        except ValueError:
                            vmax_param = None
                    cmap_param = params.get("cmap", "Blues")
                    plot_triplet(hr_var=hr_var,
                        date=date0,
                        pmm=pmm, hr=hr, lr=lr, lsm=lsm_i,
                        out_dir=sample_dir,
                        vmax=vmax_param,
                        title_suffix="(Bilinear)"
                    )
                    plotted += 1
                except Exception as e:
                    logger.warning(f"[Bilinear][plot] Failed to plot sample {date0}: {e}")

            if n_days % 200 == 0:
                logger.info(f"[Bilinear] Saved {n_days} days so far...")
    logger.info(f"[Bilinear] Processed {n_days} days for split '{split_name}'")
    logger.info(f"[Bilinear] Outputs saved to: {out_root}")
    manifest = {
        "baseline": "bilinear",
        "split": split_name,
        "num_days": n_days,
        "hr_var": cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var', 'prcp')),
        "lr_key": adapter.lr_key,
        "x_channels": adapter.extra_channels,
    }
    if not quicklook:
        (out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    return {"manifest": manifest, "out_root": out_root, "results": results}

        