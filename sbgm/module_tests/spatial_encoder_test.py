"""Spatial ContextEncoder unit test.

Run (pure unit test with random tensors):
  python -m sbgm.module_tests.spatial_encoder_test

Run (integration test using a config + dataloader batch):
  python -m sbgm.module_tests.spatial_encoder_test --config sbgm/config/paper2/P0.yaml

What it checks:
  - ContextEncoder forward pass runs
  - Output shape is [B, c_out, H_hr, W_hr]
  - Gradients flow
  - Variable-ID FiLM path works (var_ids as [V] and [B,T,V])
"""

import argparse
import os
import sys
import time

import torch
from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt
import itertools

from sbgm.utils import load_config, extract_samples
from sbgm.training_utils import get_dataloader, get_model, get_optimizer, get_scheduler
from sbgm.training import TrainingPipeline_general
from sbgm.score_unet import marginal_prob_std_fn, diffusion_coeff_fn

from sbgm.score_unet import ContextEncoder


# Register ${env:VAR} resolver for OmegaConf before loading configs
def _register_env_resolver() -> None:
    """Allow configs to use ${env:VAR} (and ${env:VAR,default})."""
    try:
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default="": os.environ.get(str(key), default),
            replace=True,
        )
    except Exception:
        pass


def _load_cfg(path: str) -> dict:
    _register_env_resolver()
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True) # type: ignore


def _require_key(d: dict, k: str) -> None:
    if k not in d:
        raise KeyError(f"Expected key '{k}' in dict. Available keys: {list(d.keys())}")


def _squeeze_channel(x: torch.Tensor) -> torch.Tensor:
    """Dataset tensors are typically [B,1,H,W]. Convert to [B,H,W] for ContextEncoder."""
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}")


# --- Helper: convert tensor to 2D numpy array for plotting
def _to_numpy_2d(x: torch.Tensor) -> np.ndarray:
    """Convert tensor to a 2D numpy array for plotting."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    x = x.detach().cpu()
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[0, 0]
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    elif x.ndim == 2:
        pass
    else:
        # fall back: take first element(s)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])[0]
    return x.numpy()


# Print environment configuration for integration test
def _print_env() -> None:
    print("Environment configured:")
    print(f"DATA_DIR={os.environ.get('DATA_DIR','')}")
    print(f"RUN_ROOT={os.environ.get('RUN_ROOT','')}")
    print(f"STATS_LOAD_DIR={os.environ.get('STATS_LOAD_DIR','')}")
    print(f"CWD={os.getcwd()}")


def _run_pure_unit_test(device: str = "cpu") -> None:
    torch.manual_seed(0)

    B, T, V = 2, 1, 3
    H_lr, W_lr = 589, 789
    H_tgt, W_tgt = 128, 128

    enc = ContextEncoder(
        num_vars=V,
        c_in=1,
        c_out=32,
        base_channels=16,
        depth=3,
        target_size=(H_tgt, W_tgt),
    ).to(device)

    x = torch.randn(B, T, V, H_lr, W_lr, device=device)

    # 1) var_ids as [V]
    y = enc(x)
    assert tuple(y.shape) == (B, 32, H_tgt, W_tgt), f"Unexpected output shape: {tuple(y.shape)}"

    # 2) var_ids as [B,T,V]
    var_ids = torch.arange(V, device=device).view(1, 1, V).expand(B, T, V)
    y2 = enc(x, var_ids=var_ids)
    assert tuple(y2.shape) == (B, 32, H_tgt, W_tgt), f"Unexpected output shape (BT V ids): {tuple(y2.shape)}"

    # Gradient check
    (y2.mean()).backward()
    has_grad = any(p.grad is not None for p in enc.parameters())
    assert has_grad, "Expected gradients on encoder parameters"

    print("[unit] Pure unit test passed: forward/shape/grads OK")


def _run_integration_test(cfg_path: str) -> None:
    from sbgm.training_utils import get_dataloader

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _load_cfg(cfg_path)
    _print_env()

    # Force single-worker loading for macOS spawn friendliness
    cfg.setdefault("data_handling", {})
    cfg["data_handling"]["num_workers"] = 0
    cfg["data_handling"]["pin_memory"] = False

    # If scaling is enabled but stats dir is missing, disable scaling for this test.
    stats_dir = ((cfg.get("paths", {}) or {}).get("stats_load_dir", "") or "").strip()
    scaling_enabled = bool((cfg.get("transforms", {}) or {}).get("scaling", False))
    if scaling_enabled and stats_dir == "":
        cfg.setdefault("transforms", {})
        cfg["transforms"]["scaling"] = False
        print("[integration] transforms.scaling=True but paths.stats_load_dir is empty; disabling scaling for this integration test.")

    train_loader, _, _ = get_dataloader(cfg, verbose=True)
    batch = next(iter(train_loader))

    # --- Figure out LR context size and HR size
    hr_size = tuple(cfg["highres"]["data_size"])

    paper2 = cfg.get("paper2", {}) or {}
    spatial = paper2.get("spatial_context", {}) or {}
    mode = str(spatial.get("mode", ""))
    if mode == "large_domain":
        lr_ctx = spatial.get("lr_context_size", None)
        if lr_ctx is None:
            raise ValueError("paper2.spatial_context.mode=large_domain but lr_context_size missing")
        lr_size = tuple(lr_ctx)
    else:
        lr_size = tuple(cfg["lowres"]["data_size"]) if cfg["lowres"].get("data_size", None) is not None else hr_size

    # --- Build ContextEncoder from config
    enc_cfg = spatial.get("encoder", {}) or {}
    enabled = bool(enc_cfg.get("enabled", True))
    if not enabled:
        raise ValueError("paper2.spatial_context.encoder.enabled is False; nothing to test")

    c_out = int(enc_cfg.get("c_out", 32))
    depth = int(enc_cfg.get("depth", 3))
    base_ch = int(enc_cfg.get("base_channels", 16))

    lr_vars = list(cfg["lowres"]["condition_variables"])
    V = len(lr_vars)

    enc = ContextEncoder(
        num_vars=V,
        c_in=1,
        c_out=c_out,
        base_channels=base_ch,
        depth=depth,
        target_size=tuple(hr_size),
    )

    # --- Build input tensor x: [B,T,V,H_lr,W_lr]
    # Dataset provides each LR var as [B,1,H_lr,W_lr] in large_domain mode.
    xs = []
    for v in lr_vars:
        k = f"{v}_lr"
        _require_key(batch, k)
        t = batch[k]
        if not torch.is_tensor(t):
            raise TypeError(f"{k} is not tensor: {type(t)}")
        # Sanity: spatial dims
        if t.ndim != 4:
            raise ValueError(f"Expected {k} to be [B,1,H,W], got {tuple(t.shape)}")
        if tuple(t.shape[-2:]) != tuple(lr_size):
            raise ValueError(f"{k} spatial size mismatch: got {tuple(t.shape[-2:])}, expected {tuple(lr_size)}")
        xs.append(_squeeze_channel(t))  # [B,H,W]

    # Stack variables -> [B,V,H,W]
    x_bvhw = torch.stack(xs, dim=1)

    # Temporal handling: start with single-day [B,1,V,H,W]
    x = x_bvhw.unsqueeze(1)

    # --- Forward
    with torch.no_grad():
        ctx = enc(x)

    assert ctx.ndim == 4, f"Expected ctx [B,C,H,W], got {tuple(ctx.shape)}"
    B = x.shape[0]
    assert tuple(ctx.shape) == (B, c_out, hr_size[0], hr_size[1]), (
        f"Unexpected ctx shape: got {tuple(ctx.shape)} expected {(B, c_out, hr_size[0], hr_size[1])}"
    )

    print("[integration] ContextEncoder output shape OK:", tuple(ctx.shape))
    print("             lr_vars:", lr_vars)
    print("             lr_size:", lr_size, "hr_size:", hr_size)
    print("ctx mean:", ctx.mean().item(), "ctx std:", ctx.std().item())

    # --- Plot dump (debug): save LR context + encoded context maps
    run_root = os.environ.get("RUN_ROOT", "").strip()
    log_root = os.environ.get("LOG_DIR", "").strip()
    if log_root == "" and run_root != "":
        log_root = os.path.join(run_root, "logs")

    if log_root != "":
        out_dir = os.path.join(log_root, "spatial_encoder_test")
        os.makedirs(out_dir, exist_ok=True)

        # Take sample 0 for visualization
        prcp0 = _to_numpy_2d(batch[f"{lr_vars[0]}_lr"][0:1])  # [1,1,H,W] -> [H,W]
        temp0 = _to_numpy_2d(batch[f"{lr_vars[1]}_lr"][0:1]) if len(lr_vars) > 1 else None

        ctx0 = ctx[0].detach().cpu()  # [C,H,W]
        ctx_mean = ctx0.mean(dim=0).numpy()
        ctx_ch0 = ctx0[0].numpy()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

        im0 = axs[0, 0].imshow(prcp0, origin="lower", interpolation="nearest")
        axs[0, 0].set_title(f"LR {lr_vars[0]} (sample0)")
        fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        if temp0 is not None:
            im1 = axs[0, 1].imshow(temp0, origin="lower", interpolation="nearest")
            axs[0, 1].set_title(f"LR {lr_vars[1]} (sample0)")
            fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
        else:
            axs[0, 1].axis("off")

        im2 = axs[1, 0].imshow(ctx_mean, origin="lower", interpolation="nearest")
        axs[1, 0].set_title("ContextEncoder output: channel-mean")
        fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im3 = axs[1, 1].imshow(ctx_ch0, origin="lower", interpolation="nearest")
        axs[1, 1].set_title("ContextEncoder output: channel 0")
        fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

        out_path = os.path.join(out_dir, "spatial_encoder_test_sample0.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved plot: {out_path}")
    else:
        print("[integration] RUN_ROOT/LOG_DIR not set; skipping plot dump")

def _run_smoke_train(
    cfg_path: str,
    *,
    epochs: int = 2,
    max_train_batches: int = 5,
    max_val_batches: int = 2,
    device: str = "cpu",
):
    """
    Tiny end-to-end smoke test:
      - builds dataloaders from cfg
      - builds model + optimizer via shared utilities
      - runs a few train + val batches (EDM + context encoder path)
      - runs a single inference-style loss call under inference_mode

    This is NOT a benchmark. It only checks wiring + shapes + no runtime errors.
    """

    cfg = _load_cfg(cfg_path)

    # -------------------------------------------------
    # Smoke-test overrides (keep it lightweight on laptop)
    # -------------------------------------------------
    # Ensure dicts exist
    cfg.setdefault("training", {})
    cfg.setdefault("data_handling", {})
    cfg.setdefault("diagnostics", {})
    cfg.setdefault("rain_gate", {})

    # Tiny batches: context encoder runs on large LR domains and can OOM with B=16
    cfg["training"]["batch_size"] = int(cfg["training"].get("batch_size", 16))
    cfg["training"]["batch_size"] = min(cfg["training"]["batch_size"], 2)

    # If you have a separate validation batch size key, keep it tiny too
    if "eval_batch_size" in cfg["training"]:
        cfg["training"]["eval_batch_size"] = min(int(cfg["training"]["eval_batch_size"]), 2)

    # Deterministic, single-process loading
    cfg["data_handling"]["num_workers"] = 0
    cfg["data_handling"]["pin_memory"] = False
    cfg["data_handling"]["persistent_workers"] = False

    # Disable heavy per-batch diagnostics for the smoke run
    cfg["diagnostics"]["per_batch_stats"] = False

    # Disable rain gate for smoke train unless explicitly enabled
    cfg["rain_gate"]["enabled"] = bool(cfg["rain_gate"].get("enabled", False)) and False

    # Keep CPU thread usage modest (helps macOS stability)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    print(f"Data dir: {cfg.get('paths', {}).get('data_dir', None)}")
    # Force fast, deterministic loader for a smoke test
    try:
        cfg["data_handling"]["num_workers"] = 0 # type: ignore
    except Exception:
        pass
    os.environ["SLURM_CPUS_PER_TASK"] = "0"

    dev = torch.device(device)

    train_loader, val_loader, _gen_loader = get_dataloader(cfg, verbose=True)

    # Build model/optimizer/scheduler the same way as a real run
    model, _ckpt_path, _ckpt_name = get_model(cfg)
    model = model.to(dev)
    optimizer = get_optimizer(cfg, model)

    print("[smoke] Model class:", model.__class__.__name__)

    lr_scheduler_type = cfg.get("training", {}).get("lr_scheduler", None) # type: ignore
    scheduler = get_scheduler(cfg, optimizer) if lr_scheduler_type is not None else None

    pipe = TrainingPipeline_general(
        model=model,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        optimizer=optimizer,
        device=dev,
        lr_scheduler=scheduler,
        cfg=cfg,
    )

    # --- Train/val a couple epochs, but only a few batches
    for ep in range(1, epochs + 1):
        t0 = time.time()
        b = next(iter(train_loader))
        print("first batch fetch seconds:", time.time() - t0)
        train_iter = itertools.islice(iter(train_loader), max_train_batches)
        pipe.train_batches(
            train_iter,
            epochs=epochs,
            current_epoch=ep,
            verbose=True,
            use_mixed_precision=False,
        )

        val_iter = itertools.islice(iter(val_loader), max_val_batches)
        pipe.validate_batches(
            val_iter,
            epochs=epochs,
            current_epoch=ep,
            verbose=True,
        )

    # --- Inference-style check: build cond the same way the pipeline does, then call loss
    batch = next(iter(train_loader))

    x, y, _cond_images_legacy, lsm_hr, lsm, sdf, topo, _hr_points, _lr_points = extract_samples(batch, dev)

    # IMPORTANT: exercise the context encoder path (context_only/context_plus_local)
    cond_img = pipe._build_cond_img(batch)

    # If your extract_samples returns topo_hr separately, prefer it here.
    topo_cond = None
    if isinstance(batch, dict) and ("topo_hr" in batch) and (batch["topo_hr"] is not None):
        topo_cond = batch["topo_hr"].to(dev)
    else:
        topo_cond = topo

    lsm_cond = lsm_hr if lsm_hr is not None else lsm

    lr_ups = None
    edm_on = bool(cfg.get("edm", {}).get("enabled", False)) # type: ignore
    if edm_on and bool(cfg.get("edm", {}).get("predict_residual", False)): # type: ignore
        # Baseline extraction in your pipeline expects the *local* LR channel to exist in the batch.
        # It uses the pipeline's cond-channel map, so we just pass cond_img and let it slice.
        lr_ups = pipe._build_lr_ups_baseline(cond_img if cond_img is not None else _cond_images_legacy)

    with torch.inference_mode():
        _ = pipe.loss_fn(
            pipe.model,
            x,
            y=y,
            cond_img=cond_img,
            lsm_cond=lsm_cond,
            topo_cond=topo_cond,
            sdf_cond=sdf,
            lr_ups=lr_ups,
            pixel_weight_map=None,
        )

    print("[smoke] Training+validation+inference smoke test completed OK")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.environ.get("SPATIAL_ENCODER_CONFIG", None),
        help="Optional YAML config path. If provided, runs a dataloader integration test.",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for the pure unit test.",
    )
    ap.add_argument(
        "--only-integration",
        action="store_true",
        help="If set, skip the pure unit test and only run the integration test.",
    )

    ap.add_argument("--smoke-train", action="store_true", help="Run tiny end-to-end train/val + inference smoke test")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-batches", type=int, default=5)
    ap.add_argument("--max-val-batches", type=int, default=2)
    args = ap.parse_args()

    # 1) Pure unit test (optional)
    if not args.only_integration:
        dev = args.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        _run_pure_unit_test(device=dev)

    # 2) Smoke train (optional)
    if args.smoke_train:
        if args.config is None:
            raise ValueError("--smoke-train was set but no --config (or $SPATIAL_ENCODER_CONFIG) was provided")
        _run_smoke_train(
            args.config,
            epochs=int(args.epochs),
            max_train_batches=int(args.max_train_batches),
            max_val_batches=int(args.max_val_batches),
            device=str(args.device),
        )
        return

    # 3) Integration test
    if args.config is not None:
        _run_integration_test(args.config)
    else:
        if args.only_integration:
            raise ValueError("--only-integration was set but no --config (or $SPATIAL_ENCODER_CONFIG) was provided")


if __name__ == "__main__":
    main()