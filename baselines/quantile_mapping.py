# sbgm/baselines/quantile_mapping.py
"""
    Core: Do CDF matching form LR_upsampled to HR truthm on land pixels only, with a wet-day threshold and piecewise-linear monotone mapping. Fit once on training, apply on val/test.

    Fitting choices (simple and defensible):
        - Scope: default monthly-global-land (all land pixels pooled per month). Keeps memory light and avoids noisy per-pixel fits; also respects seasonal cycle.
        - Wet-day handling: estimate wet-day frequency on LR_upsampled and HR truth; use a threshold (e.g. 0.1 mm/day). Below threshold treat as dry (0 mm/day).
        - Quantiles: fixed grid. Include 0 exactly to anchor spike at 0
        - Tails: beyond the largest fitted quantile, extrapolate linearly in log space (clamped to >= 0). Conservative and avoids crazy blow-ups on extremes.
        - Monotonicity: enforce strictly increasing target quantiles (np.maximum.accumulate). Avoids pathological inversions.
"""
from __future__ import annotations
import json 
from typing import Optional
import numpy as np
from pathlib import Path
import torch
import logging

from baselines.plotting import plotting_enabled, plotting_params, resolve_samples_dir, plot_triplet
from scor_dm.plotting_utils import get_cmap_for_variable

logger = logging.getLogger(__name__)

class QuantileMapper:
    def __init__(self, q_grid, wet_threshold=0.1, tail="linear-log"):
        self.q = np.asarray(q_grid)
        self.wet = wet_threshold
        self.tail = tail  # "linear-log" or "none"
        # dict[(month)] = {"qlr": ..., "qhr": ..., "p_wet_lr": ..., "p_wet_hr": ...}
        self.tables = {}
        logger.info(f"[QM] Initialized: q_levels: {len(self.q)}, wet-threshold: {self.wet}, tail-extrapolation: {self.tail}")

    @staticmethod
    def _empirical_quantiles(x, q):
        """
            Compute empirical quantiles for given data and quantile levels.
            Handles wet-day thresholding, ignores nans.
        """
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        x = np.clip(x, 0, None)  # No negative values
        if x.size == 0:
            return np.zeros_like(q)
        return np.quantile(x, q, method='linear') # type: ignore

    def fit_month(self, month, lr_up_land, hr_land):
        """
            lr_up_land, hr_land: 1D arrays with land-only pixels from all training days of given month.
        """
        # wet-day masks
        lr_wet = lr_up_land > self.wet
        hr_wet = hr_land > self.wet

        p_wet_lr = lr_wet.mean() if lr_wet.size else 0.0
        p_wet_hr = hr_wet.mean() if hr_wet.size else 0.0

        # condition on wet days for continuous part
        lr_vals = lr_up_land[lr_wet] 
        hr_vals = hr_land[hr_wet]

        qlr = self._empirical_quantiles(lr_vals, self.q) if lr_vals.size else np.zeros_like(self.q)
        qhr = self._empirical_quantiles(hr_vals, self.q) if hr_vals.size else np.zeros_like(self.q)

        # enforce monotonicity and anchor zeros
        qlr = np.maximum.accumulate(qlr)
        qhr = np.maximum.accumulate(qhr)
        qlr[0] = 0.0
        qhr[0] = 0.0

        self.tables[int(month)] = dict(qlr=qlr, qhr=qhr, p_wet_lr=p_wet_lr, p_wet_hr=p_wet_hr)

    def fit(self, train_loader, land_mask_key_present=True):
        # accumulate per-month land-only vectors
        store = {m: {"lr": [], "hr": []} for m in range(1,13)}

        for b_idx, batch in enumerate(train_loader):
            lr_up = batch.lr_up.numpy()              # [B,1,H,W]
            y     = batch.y.numpy()                  # [B,1,H,W]
            lsm   = batch.lsm.numpy().astype(bool)   # [B,1,H,W]
            months= batch.month.numpy()

            B = lr_up.shape[0]
            if (b_idx + 1) % 20 == 0:
                logger.info(f"[QM][fit] Processed {b_idx+1} batches, total samples: {(b_idx+1)*B}")
            for i in range(B):
                m = int(months[i])
                land = lsm[i,0]
                lr_i = lr_up[i,0][land].ravel()
                hr_i = y[i,0][land].ravel()
                store[m]["lr"].append(lr_i)
                store[m]["hr"].append(hr_i)

        for m in range(1,13):
            lr_cat = np.concatenate(store[m]["lr"]) if store[m]["lr"] else np.array([])
            hr_cat = np.concatenate(store[m]["hr"]) if store[m]["hr"] else np.array([])
            self.fit_month(m, lr_cat, hr_cat)
        
        months_fit = []
        for m in range(1, 13):
            tab = self.tables.get(m, None)
            if tab is None:
                continue
            qlr, qhr = tab["qlr"], tab["qhr"]
            p_lr, p_hr = tab["p_wet_lr"], tab["p_wet_hr"]
            months_fit.append(m)
            logger.info(f"[QM][fit] Month {m:02d} | p_wet_lr={p_lr:.4f}, p_wet_hr={p_hr:.4f}, q_max_lr={qlr[-1]:.3f}, q_max_hr={qhr[-1]:.3f}")
        
        logger.info(f"[QM][fit] Completed. Months fitted: {months_fit if months_fit else 'None'}")


    def _map_wet(self, x, qlr, qhr):
        # np.interp only supports scalar for 'right', so handle extrapolation manually
        x = np.asarray(x)
        interp_vals = np.interp(x, qlr, qhr, left=qhr[0], right=qhr[-1])
        # Find indices where x > qlr[-1] to apply custom extrapolation
        extrap_mask = x > qlr[-1]
        if np.any(extrap_mask):
            interp_vals[extrap_mask] = self._extrapolate(x[extrap_mask], qlr, qhr)
        return interp_vals

    def _extrapolate(self, x, qlr, qhr):
        x = np.asarray(x)
        ymax = max(qhr[-1], 0.0)
        if self.tail == "linear-log" and qlr[-2] > 0 and qhr[-2] > 0:
            x1, x2 = qlr[-2], qlr[-1]
            y1, y2 = qhr[-2], qhr[-1]
            b = (np.log(y2+1e-9) - np.log(y1+1e-9)) / (np.log(x2+1e-9) - np.log(x1+1e-9) + 1e-12)
            a = np.log(y2+1e-9) - b*np.log(x2+1e-9)
            y = np.exp(a + b*np.log(np.maximum(x, x2)+1e-9))
            return y
        return np.full_like(x, ymax)

    def transform_batch(self, lr_up_bchw, months_b, land_mask_bchw):
        B,_,H,W = lr_up_bchw.shape
        logger.info(f"[QM][apply] Transforming batch: B={B}, HxW={H}x{W}")
        out = np.zeros_like(lr_up_bchw)
        for i in range(B):
            m = int(months_b[i])
            tb = self.tables.get(m)
            if tb is None:
                tb = self.tables.get(1)
                if tb is None:
                    raise ValueError(f"No quantile mapping table found for month {m} or default month 1.")
            qlr, qhr = tb["qlr"], tb["qhr"]
            x = lr_up_bchw[i,0]
            wet = x >= self.wet
            y = np.zeros_like(x)
            y[wet] = self._map_wet(x[wet], qlr, qhr)
            land = land_mask_bchw[i,0].astype(bool)
            y[~land] = 0.0
            out[i,0] = y
        return out
    

def run_qm(cfg, adapter_train, adapter_apply, out_root, split_name: str = "test",
           quicklook: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = cfg.get('baseline', {}).get('experiment_name', 'qm_baseline')
    hr_var = cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var', 'prcp'))
    cmap = get_cmap_for_variable(hr_var)
        
    logger.info(f"[QM] Output root: {out_root}")

    if not quicklook:
        for d in ['pmm', 'lr_hr', 'lsm', 'meta']:
            (out_root / d).mkdir(parents=True, exist_ok=True)
    # plotting setup (optional)
    want_plot = plotting_enabled(cfg)
    params = plotting_params(cfg) if want_plot else {}
    plotted = 0
    sample_dir = None
    if want_plot: 
        sample_dir = resolve_samples_dir(cfg, baseline_type='qm', split=split_name)
        logger.info(f"[QM] Plotting enabled. Samples dir: {sample_dir}")

    q_grid = cfg.get('baseline', {}).get('qm', {}).get('quantiles',
              [0.00,0.01,0.02,0.05,0.10,0.20,0.40,0.60,0.80,0.90,0.95,0.98,0.99,0.995])
    wet = cfg.get('baseline', {}).get('qm', {}).get('wetday_threshold', 0.1)

    qm = QuantileMapper(q_grid, wet_threshold=wet, tail="linear-log")
    logger.info(f"[QM] Fitting on training split with wetday_treshold={wet} and {len(q_grid)} quantiles...")
    loader_train = adapter_train.make_loader(batch_size=cfg.get('batch_size', 8),
                                             shuffle=False, num_workers=cfg.get('num_workers', 4))
    qm.fit(loader_train, land_mask_key_present=True)
    months_ready = sorted([m for m, t in qm.tables.items() if t is not None])
    logger.info(f"[QM] Fit completed. Tables available for months: {months_ready if months_ready else 'None'}")

    loader_apply = adapter_apply.make_loader(batch_size=cfg.get('batch_size', 8),
                                             shuffle=False, num_workers=cfg.get('num_workers', 4))
    n_days = 0
    results = [] if quicklook else None

    for batch in loader_apply:
        dates, lr_up, y, lsm = batch.date, batch.lr_up.numpy(), batch.y.numpy(), batch.lsm.numpy()
        months = batch.month.numpy()
        logger.info(f"[QM] Applying mapping to batch of {len(dates)} samples, months: {months.tolist()}")
        yhat = qm.transform_batch(lr_up, months, lsm)  # [B,1,H,W]

        for i in range(yhat.shape[0]):
            date0 = dates[i]
            pmm = yhat[i:i+1]     # [1,1,H,W]
            hr  = y[i:i+1] if y is not None else np.empty((0,))
            lr  = lr_up[i:i+1]
            lsm_i = lsm[i:i+1]

            if quicklook and results is not None:
                results.append({"date": date0, "pmm": pmm, "hr": hr, "lr_hr": lr})
            else:
                np.savez_compressed(out_root / 'pmm' / f'{date0}.npz', pmm=pmm)
                np.savez_compressed(out_root / 'lr_hr' / f'{date0}.npz', hr=hr, lr_hr=lr)
                np.savez_compressed(out_root / 'lsm' / f'{date0}.npz', lsm=lsm_i)
            n_days += 1
            if n_days % 200 == 0:
                logger.info(f"[QM] Saved {n_days} days so far...")

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

    logger.info(f"[QM] Done. Total saved days for split '{split_name}': {n_days}")
    logger.info(f"[QM] Outputs saved to: {out_root if not quicklook else 'in-memory results'}")
    manifest = {
        "baseline": "quantile_mapping",
        "split": split_name,
        "num_days": n_days,
        "hr_var": cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var', 'prcp')),
        "lr_key": adapter_apply.lr_key,
        "q_grid": q_grid,
        "wetday_threshold": wet,
    }
    if not quicklook:
        (out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    return {"manifest": manifest, "out_root": str(out_root), "results": results or []}


