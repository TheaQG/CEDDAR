# sbgm/baselines/unet_sr/train.py
from __future__ import annotations
import numpy as np
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from baselines.unet_sr.model import TinyUNet
from baselines.plotting import plotting_enabled, plotting_params, resolve_samples_dir, plot_triplet
from torch.nn.utils import clip_grad_norm_
def _masked_mean_safe(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over t where mask==True. If mask has no True, fall back to unmasked mean.
    Returns a scalar tensor (preserves autograd).
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # broadcast mask to t
    while mask.dim() < t.dim():
        mask = mask.unsqueeze(1)
    denom = mask.sum()
    if denom == 0:
        return t.mean()
    return (t * mask).sum() / denom

from sbgm.special_transforms import build_back_transforms_from_stats

logger = logging.getLogger(__name__)

# --- light geometric augs to reduce overfitting / improve structure ---
def _random_geo_augs(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor):
    # x:[B,C,H,W], y:[B,1,H,W], m:[B,1,H,W]
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1]); y = torch.flip(y, dims=[-1]); m = torch.flip(m, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2]); y = torch.flip(y, dims=[-2]); m = torch.flip(m, dims=[-2])
    k = int(torch.randint(0, 4, ()).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1]); y = torch.rot90(y, k, dims=[-2, -1]); m = torch.rot90(m, k, dims=[-2, -1])
    return x, y, m

# --- helper to construct back-transforms (mirrors generation.py logic)
def _build_back_transforms(cfg: dict):
    """
    Build callable back-transforms:
      - 'bt_gen'     : generated → physical
      - 'bt_hr'      : HR       → physical
      - 'bt_lr_HR'   : LR (HR/HR+LR-scaled) → physical
      - 'bt_lr_LR'   : LR (LR-only scaled)  → physical
    """
    full_domain_dims_hr = cfg['highres'].get('full_domain_dims', None)
    full_domain_dims_str_hr = f"{full_domain_dims_hr[0]}x{full_domain_dims_hr[1]}" if full_domain_dims_hr is not None else "full_domain"
    crop_region_hr = cfg['highres'].get('cutout_domains', None)
    crop_region_hr_str = '_'.join(map(str, crop_region_hr)) if crop_region_hr is not None else 'no_crop'

    full_domain_dims_lr = cfg['lowres'].get('full_domain_dims', None)
    full_domain_dims_str_lr = f"{full_domain_dims_lr[0]}x{full_domain_dims_lr[1]}" if full_domain_dims_lr is not None else "full_domain"
    crop_region_lr = cfg['lowres'].get('cutout_domains', None)
    crop_region_lr_str = '_'.join(map(str, crop_region_lr)) if crop_region_lr is not None else 'no_crop'

    bt_all = build_back_transforms_from_stats(
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
        split=cfg.get('transforms', {}).get('scaling_split', 'train'),
        stats_dir_root=cfg['paths']['stats_load_dir'],
        eps=cfg.get('transforms', {}).get('prcp_eps', 0.01),
    )

    logger.info("[UNetSR][bt] keys: %s", list(bt_all.keys()))
    hr_var = cfg['highres']['variable']

    # Heuristics to pick LR inverses:
    # - HR-flavored inverse: keys that mention 'hr_lr'/'hrlr' or 'hr' along with the LR/target var
    # - LR-flavored inverse: keys that mention 'lr' (but not 'hr_lr') with the LR/target var
    
    def pick_lr(bt_dict: dict, prefer: str) -> Tuple[Optional[Callable], Optional[str]]:
        prefer = (prefer or "HR").upper()
        keys = [k for k in bt_dict.keys() if (hr_var.lower() in k.lower()) and ('lr' in k.lower())]
        if not keys:
            return None, None

        def score_hr(k: str) -> int:
            kl = k.lower()
            s = 0
            if ('hr_lr' in kl) or ('hrlr' in kl) or ('hr+lr' in kl): s += 10
            if 'hr' in kl: s += 3
            if 'lr' in kl: s += 1
            return s

        def score_lr(k: str) -> int:
            kl = k.lower()
            s = 0
            # prefer pure-lr mention without 'hr_lr'
            if 'lr' in kl: s += 3
            if ('hr_lr' in kl) or ('hrlr' in kl) or ('hr+lr' in kl): s -= 5
            if 'hr' in kl: s -= 1
            return s

        if prefer == "HR":
            cand = sorted(keys, key=lambda k: (-score_hr(k), k))
        else:
            cand = sorted(keys, key=lambda k: (-score_lr(k), k))

        for k in cand:
            fn = bt_dict.get(k, None)
            if callable(fn): return fn, k

        # fallback: any callable among keys
        for k in keys:
            fn = bt_dict.get(k, None)
            if callable(fn): return fn, k
        return None, None

    bt_gen = bt_all.get('generated', None)
    bt_hr  = bt_all.get(f"{hr_var}_hr", None)

    bt_lr_HR, key_hr = pick_lr(bt_all, "HR")
    bt_lr_LR, key_lr = pick_lr(bt_all, "LR")

    logger.info("[UNetSR][bt] chosen: bt_gen=%s  bt_hr=%s  bt_lr_HR=%s(%s)  bt_lr_LR=%s(%s)",
                'ok' if callable(bt_gen) else None,
                'ok' if callable(bt_hr) else None,
                'ok' if callable(bt_lr_HR) else None, key_hr,
                'ok' if callable(bt_lr_LR) else None, key_lr)

    return {"bt_gen": bt_gen, "bt_hr": bt_hr, "bt_lr_HR": bt_lr_HR, "bt_lr_LR": bt_lr_LR}


def evaluate(model, loader, loss_fn, device):
    """
        Evaluate model on the given data split using the provided loss function.
        Returns average loss over the split.
    """
    logger.debug("[UNetSR][eval] Starting evaluation...")
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x_in, y, lsm = batch.x_in.to(device), batch.y.to(device), batch.lsm.to(device).bool()
            yhat = model(x_in)
            perpx = loss_fn(yhat, y)  # [B,1,H,W]
            loss_t = _masked_mean_safe(perpx, lsm)  # scalar tensor
            loss = loss_t.item()
            tot += loss; cnt += 1
    model.train()
    return tot / max(cnt,1)


def save_split_outputs(model, loader, out_root: Path, device: torch.device, cfg):
    """
        Saves model predictions on the given data split to out_root in a generation-like folder structure:
            out_root/
                pmm/{date}.npz -> {'pmm': [1,1,H,W]}
                lr_hr/{date}.npz -> {'hr': [1,1,H,W], 'lr_hr': [1,1,H,W]}
                lsm/{date}.npz -> {'lsm': [1,1,H,W]}
                meta/manifest.json -> {'date': {date}, 'shape': [1,1,H,W]}
    """
    hr_var = cfg.get('highres', {}).get('variable', cfg.get('data', {}).get('var','prcp'))
    logger.info(f"[UNetSR] Saving outputs to: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / 'pmm').mkdir(parents=True, exist_ok=True)
    (out_root / 'lr_hr').mkdir(parents=True, exist_ok=True)
    (out_root / 'lsm').mkdir(parents=True, exist_ok=True)
    (out_root / 'meta').mkdir(parents=True, exist_ok=True)

    # physical-space dirs (mirror generation.py)
    (out_root / 'pmm_phys').mkdir(parents=True, exist_ok=True)
    (out_root / 'lr_hr_phys').mkdir(parents=True, exist_ok=True)

    # build back-transforms
    bt = _build_back_transforms(cfg)
    bt_gen   = bt.get("bt_gen", None)
    bt_hr    = bt.get("bt_hr",  None)
    bt_lr_HR = bt.get("bt_lr_HR", None)
    bt_lr_LR = bt.get("bt_lr_LR", None)

    # --- mapping policy driven by YAML ---
    # Channel-0 follows lr_main_var_scale (HR_LR → HR), channel-1 (if present) uses the other inverse.
    main_scale = (cfg.get('lowres', {}).get('lr_main_var_scale', 'HR') or 'HR').upper()
    if main_scale in ('HR_LR', 'HRLR', 'HR+LR'):
        main_scale = 'HR'
    dual_lr = bool(cfg.get('lowres', {}).get('dual_lr', False))

    # Which LR channel to plot if dual (0 or 1)
    plot_ch = int(cfg.get('visualization', {}).get('plot_dual_lr_channel', 0))

    logger.info("[UNetSR][save] LR policy: dual_lr=%s, lr_main_var_scale=%s, plot_ch=%d",
                dual_lr, main_scale, plot_ch)

    if bt_gen is None or bt_hr is None:
        logger.warning("[UNetSR][save] Back-transform functions missing (bt_gen=%s, bt_hr=%s). Physical outputs will be skipped.", bt_gen, bt_hr)

    # plotting setup (optional)
    want_plot = plotting_enabled(cfg)
    params = plotting_params(cfg) if want_plot else {}
    plotted = 0
    if want_plot:
        # assume test split for now; adjust if you call with different split names
        sample_dir = resolve_samples_dir(cfg, baseline_type="unet_sr", split="test")
        logger.info(f"[UNetSR][plot] Enabled. Saving examples to: {sample_dir}")
    else:
        sample_dir = None

    logger.info("[UNetSR][save] Generating predictions")
    model.eval()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            x_in, y, lsm = batch.x_in.to(device), batch.y, batch.lsm.bool()
            dates = batch.date
            yhat = model(x_in).cpu().numpy()
            y_np = y.cpu().numpy()
            lsm_np = lsm.cpu().numpy()
            lr_np  = batch.lr_up.cpu().numpy()
            
            if b_idx == 0:
                B, _, H, W = yhat.shape if isinstance(yhat, np.ndarray) else (len(dates),) + tuple(batch.y.shape[1:])
                logger.info(f"[UNetSR][save] First batch: B={len(dates)}, HxW~{batch.y.shape[-2:]}")
            for i, d in enumerate(dates):
                # --- Save model-space outputs
                np.savez_compressed(out_root / 'pmm' / f'{d}.npz', pmm=yhat[i:i+1])
                np.savez_compressed(out_root / 'lr_hr' / f'{d}.npz', hr=y_np[i:i+1], lr_hr=lr_np[i:i+1])
                np.savez_compressed(out_root / 'lsm' / f'{d}.npz', lsm=lsm_np[i:i+1])

                # Initialize possibly unbound variables
                pmm_phys_np = None
                hr_phys_np = None
                lr_phys_np = None
                lr0_phys_np = None
                lr1_phys_np = None

                # --- optionally save physical-space using back-transforms
                try:
                    if (bt_gen is not None) and (bt_hr is not None):
                        yhat_t = torch.from_numpy(yhat[i:i+1])
                        hr_t   = torch.from_numpy(y_np[i:i+1])

                        pmm_phys = bt_gen(yhat_t) if callable(bt_gen) else None
                        hr_phys  = bt_hr(hr_t)    if callable(bt_hr)  else None

                        # --- LR channels → physical ---
                        lr0_phys = lr1_phys = None
                        if lr_np is not None:
                            lr_t_full = torch.from_numpy(lr_np[i:i+1])  # [1,C,H,W]
                            C = lr_t_full.shape[1]
                            if C >= 1:
                                ch0 = lr_t_full[:, 0:1]
                                if main_scale == "HR" and callable(bt_lr_HR):
                                    lr0_phys = bt_lr_HR(ch0)
                                elif main_scale == "LR" and callable(bt_lr_LR):
                                    lr0_phys = bt_lr_LR(ch0)
                            if dual_lr and C >= 2:
                                ch1 = lr_t_full[:, 1:2]
                                # channel-1 is the complementary scale if available
                                if main_scale == "HR" and callable(bt_lr_LR):
                                    lr1_phys = bt_lr_LR(ch1)
                                elif main_scale == "LR" and callable(bt_lr_HR):
                                    lr1_phys = bt_lr_HR(ch1)
                            else:
                                # single-channel LR
                                ch = lr_t_full
                                if main_scale == "HR" and callable(bt_lr_HR):
                                    lr0_phys = bt_lr_HR(ch)
                                elif main_scale == "LR" and callable(bt_lr_LR):
                                    lr0_phys = bt_lr_LR(ch)

                        def _to_np(x):
                            if x is None: return None
                            if torch.is_tensor(x): x = x.detach().cpu().float()
                            return x.numpy()

                        pmm_phys_np = _to_np(pmm_phys)
                        hr_phys_np  = _to_np(hr_phys)
                        lr0_phys_np = _to_np(lr0_phys)
                        lr1_phys_np = _to_np(lr1_phys)

                        if pmm_phys_np is not None:
                            np.savez_compressed(out_root / 'pmm_phys' / f'{d}.npz', pmm=pmm_phys_np)

                        # --- choose canonical LR for lr_hr_phys: prefer LR-in-its-own-stats ---
                        canonical_lr_np = None
                        if dual_lr:
                            if main_scale == "HR":
                                # ch0 used HR inverse; canonical should be the complementary LR inverse
                                canonical_lr_np = lr1_phys_np if lr1_phys_np is not None else lr0_phys_np
                            else:  # main_scale == "LR"
                                canonical_lr_np = lr0_phys_np if lr0_phys_np is not None else lr1_phys_np
                        else:
                            canonical_lr_np = lr0_phys_np  # only mapping available
                        
                        # Write lr_hr_phys: hr + canonical lr + optional lr0/lr1 for debugging
                        arrs = {}
                        if hr_phys_np  is not None: arrs['hr']  = hr_phys_np
                        if canonical_lr_np is not None: arrs['lr']  = canonical_lr_np  # canonical LR for eval/plots
                        # keep explicit channels for debugging/inspection
                        if lr0_phys_np is not None: arrs['lr0'] = lr0_phys_np
                        if lr1_phys_np is not None: arrs['lr1'] = lr1_phys_np
                        if arrs:
                            np.savez_compressed(out_root / 'lr_hr_phys' / f'{d}.npz', **arrs)

                    else:
                        logger.debug(f"[UNetSR][save] Skipping physical outputs for {d} (no back-transform).")
                except Exception as e:
                    logger.warning(f"[UNetSR][save] Physical-space save failed for {d}: {e}")
                
                if (i + 1) % 200 == 0:
                    logger.info(f"[UNetSR][save] Saved {i+1} samples in current batch...")

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
                        # choose physical arrays for plotting when available; else fallback to model-space
                        pmm_plot = pmm_phys_np if (pmm_phys_np is not None) else yhat[i:i+1]
                        hr_plot  = hr_phys_np  if (hr_phys_np  is not None) else y_np[i:i+1]
                        # LR panel choice: channel selected by visualization.plot_dual_lr_channel
                        if dual_lr:
                            if plot_ch == 0 and lr0_phys_np is not None:
                                lr_plot = lr0_phys_np
                            elif plot_ch == 1 and lr1_phys_np is not None:
                                lr_plot = lr1_phys_np
                            else:
                                # fallback to whatever exists, then to model-space
                                lr_plot = lr0_phys_np if lr0_phys_np is not None else (lr1_phys_np if lr1_phys_np is not None else lr_np[i:i+1])
                        else:
                            lr_plot = lr0_phys_np if lr0_phys_np is not None else lr_np[i:i+1]
                        plot_triplet(hr_var=hr_var,
                            date=d,
                            pmm=pmm_plot, hr=hr_plot, lr=lr_plot, lsm=lsm_np[i:i+1],
                            out_dir=sample_dir,
                            vmax=vmax_param,
                            title_suffix="(UNet-SR)"
                        )
                        plotted += 1
                    except Exception as e:
                        logger.warning(f"[UNetSR][plot] Failed to plot sample {d}: {e}")

    logger.info("[UNetSR][save] Completed writing predictions.")


def run_unet_sr(cfg, adapter_train, adapter_val, adapter_test, out_root: Path):
    logger.info("[UNetSR] Initializing training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hp = cfg.get('baseline', {}).get('unet_sr', {})
    in_ch   = hp.get('in_channels', 3)
    out_ch  = hp.get('out_channels', 1)
    width   = hp.get('width', 64)
    depth   = hp.get('depth', 4)
    act     = hp.get('act', 'SiLU')
    residual= hp.get('residual', True)
    loss_nm = hp.get('loss', 'L1')
    lr      = hp.get('lr', 0.002)
    bs      = hp.get('batch_size', 16)
    steps   = hp.get('max_steps', 60000)
    amp     = hp.get('amp', True)

    n_workers = cfg.get('num_workers', 4)

    logger.info(f"[UNetSR] Hyperparameters: in_ch={in_ch}, width={width}, depth={depth}, residual={residual}, loss={loss_nm}, lr={lr}, bs={bs}, steps={steps}, amp={amp}")

    # loaders
    tr_loader = adapter_train.make_loader(batch_size=bs, shuffle=True,  num_workers=n_workers)
    va_loader = adapter_val.make_loader(batch_size=bs, shuffle=False, num_workers=n_workers)
    te_loader = adapter_test.make_loader(batch_size=bs, shuffle=False, num_workers=n_workers)
    logger.info(f"[UNetSR] Data loaders ready (bs={bs}), num_workers={n_workers}")

    # === Infer input channels from a probe batch BEFORE creating model ===
    try:
        probe = next(iter(tr_loader))
        inferred_in_ch = int(probe.x_in.shape[1])
        if inferred_in_ch != in_ch:
            logger.warning(f"[UNetSR] Warning: inferred in_ch={inferred_in_ch} from data, but config specifies in_ch={in_ch}. Using inferred value.")
            in_ch = inferred_in_ch
        logger.debug(f"[UNetSR] Inferred input channels from data: in_ch={in_ch}")
    except Exception as e:
        logger.error(f"[UNetSR] Failed to infer input channels from data: {e}. Proceeding with config in_ch={in_ch}.")
    
    model = TinyUNet(in_ch, out_ch, width=width, depth=depth, act=act, residual=residual).to(device)

    # loss/opt
    loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0) if loss_nm == 'L1' else nn.MSELoss(reduction='none')
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = GradScaler(enabled=amp)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=max(lr * 0.1, 1e-5))

    # train loop
    step, best = 0, 1e9
    model.train()
    while step < steps:
        for batch in tr_loader:
            x_in, y, lsm = batch.x_in.to(device), batch.y.to(device), batch.lsm.to(device).bool()
            # Skip batches with no valid land pixels
            if not lsm.any():
                logger.debug("[UNetSR][train] Skipping batch with zero valid mask.")
                continue
            # simple on-the-fly flips/rotations (keeps geophysical coherence)
            x_in, y, lsm = _random_geo_augs(x_in, y, lsm)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                yhat = model(x_in)
                perpx = loss_fn(yhat, y)  # [B,1,H,W]
                loss = _masked_mean_safe(perpx, lsm)

            # Guard against NaN/Inf losses (e.g., empty mask batches)
            if not torch.isfinite(loss):
                logger.warning("[UNetSR][train] Non-finite loss detected (loss=%s). Skipping step.", loss.item() if loss.numel() == 1 else str(loss))
                continue

            scaler.scale(loss).backward()

            # Optional: clip to prevent rare exploding gradients with AMP
            if any(p.requires_grad for p in model.parameters()):
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(opt); scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                logger.info(f"[UNetSR] Progress: step {step}/{steps}")
            if step % 20 == 0:
                logger.info(f"[{step}] train loss={loss.item():.4f}")
            if step % 200 == 0:
                v = evaluate(model, va_loader, loss_fn, device)
                logger.info(f"[{step}] val loss={v:.4f}")
                if v < best:
                    best = v
            if step >= steps:
                break

    logger.info("[UNetSR] Training complete. Saving predictions on test set...")
    run_name = cfg.get('experiment_name', 'baseline_unet_sr')

    logger.info(f"[UNetSR] Output base dir: {out_root}")
    # save_split_outputs(model, va_loader, out_root / 'val',  device)
    save_split_outputs(model, te_loader, out_root, device, cfg)
    logger.info("[UNetSR] Test split predictions saved.")

    (out_root / 'meta').mkdir(parents=True, exist_ok=True)
    manifest = {
        "baseline": "unet_sr",
        "steps": steps,
        "best_val": best,
        "in_channels": in_ch,
        "width": width, "depth": depth, "residual": residual,
    }
    (out_root / 'meta' / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return {"manifest": manifest, "out_root": str(out_root)}