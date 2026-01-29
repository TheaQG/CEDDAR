# sbgm/cli/dry_run.py
# from __future__ import annotations
# import logging
# import torch
# from omegaconf import OmegaConf

# from sbgm.logging_utils import log_banner
# from sbgm.training_utils import get_model, get_loss_fn
# from sbgm.data_modules import get_train_valid_dataloaders  # whatever you call it
# from sbgm.score_sampling import edm_sampler, pc_sampler     # reuse your existing API
# from sbgm.utils import get_device_from_cfg

# logger = logging.getLogger(__name__)

# def _summarize_batch(batch, land_only=False):
#     """Log quick stats to catch NaNs/shape mixups."""
#     keys = ["highres", "lowres", "lsm", "topo", "seasons"]
#     for k in keys:
#         if k in batch and batch[k] is not None and torch.is_tensor(batch[k]):
#             x = batch[k]
#             with torch.no_grad():
#                 n_nan = torch.isnan(x).sum().item()
#                 n_inf = torch.isinf(x).sum().item()
#                 logger.info(f"[batch] {k:>7s}: shape={tuple(x.shape)} dtype={x.dtype} "
#                             f"min={x.min().item():.3g} max={x.max().item():.3g} "
#                             f"nan={n_nan} inf={n_inf}")
#     if land_only and "lsm" in batch and "highres" in batch:
#         m = (batch["lsm"] > 0.5).to(batch["highres"].dtype)
#         frac_land = m.mean().item()
#         logger.info(f"[batch] land fraction = {frac_land:.3f}")

# def _param_report(model: torch.nn.Module):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     logger.info(f"[model] params total={total:,} trainable={trainable:,}")

# def _gpu_report():
#     if torch.cuda.is_available():
#         dev = torch.cuda.current_device()
#         logger.info(f"[cuda] device={torch.cuda.get_device_name(dev)} "
#                     f"capability={torch.cuda.get_device_capability(dev)} "
#                     f"mem_alloc={torch.cuda.memory_allocated()/1e9:.2f} GB "
#                     f"mem_reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")
#     else:
#         logger.info("[cuda] not available; running on CPU")

# def run(cfg):
#     """
#     Dry run = full setup (config->dataloaders->model->loss->one forward->1-2 sampling steps)
#     but NO optimizer/backprop/epochs.
#     """
#     log_banner("DRY RUN: SETUP")

#     # Optionally make a tiny override so dry run stays quick:
#     dcfg = OmegaConf.merge(cfg, {
#         "training": {"batch_size": max(1, int(cfg.training.batch_size) // 4)},
#         "monitoring": {"edm_metrics_every": 10},
#         "evaluation": {"n_steps": 2, "n_gen_samples": 1, "batch_size": 1},
#         "visualization": {"create_figs": False, "save_figs": False}
#     })

#     device = get_device_from_cfg(dcfg)
#     logger.info(f"[cfg] device={device} | edm.enabled={dcfg.edm.enabled} | "
#                 f"sampler={dcfg.sampler.sampler_type}")

#     # --------------------
#     # Dataloaders
#     # --------------------
#     log_banner("DRY RUN: BUILD DATA")
#     train_loader, valid_loader, gen_loader = get_train_valid_dataloaders(dcfg)
#     logger.info(f"[data] train batches={len(train_loader)} "
#                 f"valid batches={len(valid_loader)} gen batches={len(gen_loader)}")

#     # Pull one small batch
#     batch = next(iter(train_loader))
#     _summarize_batch(batch, land_only=True)

#     # --------------------
#     # Model & Loss
#     # --------------------
#     log_banner("DRY RUN: BUILD MODEL")
#     model, marginal_prob_std_fn = get_model(dcfg)  # your helper should return (model, fn/None)
#     model.to(device).eval()
#     _param_report(model)
#     _gpu_report()

#     loss_fn = get_loss_fn(dcfg, marginal_prob_std_fn)
#     logger.info(f"[loss] using {loss_fn.__class__.__name__}")

#     # --------------------
#     # One forward (no grad)
#     # --------------------
#     log_banner("DRY RUN: FORWARD PASS")
#     x = batch["highres"].to(device)
#     seasons = batch.get("seasons", None)
#     cond_img = batch.get("lowres", None)
#     lsm = batch.get("lsm", None)
#     topo = batch.get("topo", None)
#     sdf = batch.get("sdf", None)

#     # Move tensors to device if present
#     for name, arr in [("seasons", seasons), ("lowres", cond_img),
#                       ("lsm", lsm), ("topo", topo), ("sdf", sdf)]:
#         if torch.is_tensor(arr):
#             batch[name] = arr.to(device)

#     with torch.no_grad():
#         # Call loss.forward() in its “no-train” usage to check shapes & numerics.
#         # For DSMLoss, internally samples t; for EDMLoss, samples sigma.
#         try:
#             loss_val = loss_fn(model,
#                                x,
#                                y=batch.get("seasons"),
#                                cond_img=batch.get("lowres"),
#                                lsm_cond=batch.get("lsm"),
#                                topo_cond=batch.get("topo"),
#                                sdf_cond=batch.get("sdf"))
#             logger.info(f"[forward] loss computed OK: {float(loss_val):.4g}")
#         except TypeError:
#             # Some DSMLoss variants require marginal_prob_std_fn explicitly
#             loss_val = loss_fn(model,
#                                x,
#                                marginal_prob_std_fn,
#                                y=batch.get("seasons"),
#                                cond_img=batch.get("lowres"),
#                                lsm_cond=batch.get("lsm"),
#                                topo_cond=batch.get("topo"),
#                                sdf_cond=batch.get("sdf"))
#             logger.info(f"[forward] loss computed OK: {float(loss_val):.4g}")

#     _gpu_report()

#     # --------------------
#     # Tiny sampling smoke test
#     # --------------------
#     log_banner("DRY RUN: SAMPLER SMOKE TEST")
#     try:
#         with torch.no_grad():
#             if dcfg.sampler.sampler_type == "edm_sampler":
#                 gen = edm_sampler(
#                     model,
#                     (1, x.shape[1], *dcfg.highres.data_size),
#                     num_steps=int(dcfg.evaluation.n_steps),
#                     cfg=dcfg,
#                     device=device,
#                     # conditionals (use current batch)
#                     y=batch.get("seasons"),
#                     cond_img=batch.get("lowres"),
#                     lsm_cond=batch.get("lsm"),
#                     topo_cond=batch.get("topo"),
#                     predict_residual=bool(getattr(dcfg, "predict_residual", False)),
#                     lr_ups=batch.get("lr_ups", None),
#                 )
#             else:
#                 # fallback to your VE/PC sampler with 2 steps
#                 gen = pc_sampler(
#                     model,
#                     (1, x.shape[1], *dcfg.highres.data_size),
#                     num_steps=int(dcfg.evaluation.n_steps),
#                     cfg=dcfg,
#                     device=device,
#                     y=batch.get("seasons"),
#                     cond_img=batch.get("lowres"),
#                     lsm_cond=batch.get("lsm"),
#                     topo_cond=batch.get("topo"),
#                 )
#         logger.info(f"[sampler] success. gen.shape={tuple(gen.shape)} "
#                     f"min={gen.min().item():.3g} max={gen.max().item():.3g}")
#     except Exception as e:
#         logger.exception(f"[sampler] FAILED: {e}")
#         raise

#     log_banner("DRY RUN: OK")