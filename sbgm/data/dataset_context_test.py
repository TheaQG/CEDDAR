"""Quick sanity test for spatial context / crop alignment.

Run:
  python -m sbgm.data.dataset_context_test --config sbgm/config/paper2/P0.yaml

This script loads one batch and prints/validates tensor shapes and crop points.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Avoid GUI backend for potential plotting (if needed)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from omegaconf import OmegaConf


# Register ${env:VAR} resolver for OmegaConf before loading configs
def _register_env_resolver() -> None:
    """Allow configs to use ${env:VAR} (and ${env:VAR,default}).

    OmegaConf ships with ${oc.env:VAR}, but this repo historically used ${env:VAR}.
    Registering keeps backward-compat with existing YAMLs.
    """
    try:
        OmegaConf.register_new_resolver(
            "env",
            lambda key, default=None: os.environ.get(str(key), default),
            replace=True,
        )
    except Exception:
        # Safe fallback: if already registered in this process
        pass

from sbgm.training_utils import get_dataloader
from sbgm.plotting_utils import plot_sample


def _load_cfg(path: str) -> dict:
    _register_env_resolver()
    cfg = OmegaConf.load(path)
    # Resolve ${env:VAR} and other interpolations
    return OmegaConf.to_container(cfg, resolve=True)


def _shape(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    # numpy arrays may appear in a few optional keys
    try:
        return tuple(x.shape)
    except Exception:
        return type(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sbgm/config/paper2/P0.yaml", help="Path to YAML config")
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _load_cfg(cfg_path)

    # Force single-worker loading for this test to avoid macOS spawn pickling issues
    # (Dataset may contain non-picklable callables such as transforms.Lambda).
    cfg.setdefault("data_handling", {})
    cfg["data_handling"]["num_workers"] = 0
    cfg["data_handling"]["pin_memory"] = False

    train_loader, val_loader, gen_loader = get_dataloader(cfg, verbose=True)

    batch = next(iter(train_loader))



    # ------------------------------------------------------------
    # Plot sample 0 using the repo's standard plotting utility
    # ------------------------------------------------------------
    out_dir = cfg.get("paths", {}).get("log_dir", None) or os.getcwd()
    fig_dir = os.path.join(out_dir, "dataset_context_test")
    os.makedirs(fig_dir, exist_ok=True)

    def _unbatch_sample(batch_dict: dict, i: int = 0) -> dict:
        """Convert a collated batch dict into a single-sample dict compatible with plot_sample()."""
        sample = {}
        for k, v in batch_dict.items():
            # Dates are usually list[str]
            if k == "date" and isinstance(v, (list, tuple)):
                sample[k] = v[i]
                continue

            # hr_points/lr_points often come out as [tensor(B), tensor(B), tensor(B), tensor(B)]
            if k in ("hr_points", "lr_points") and isinstance(v, (list, tuple)) and len(v) == 4 and all(torch.is_tensor(x) for x in v):
                sample[k] = [int(v[0][i].item()), int(v[1][i].item()), int(v[2][i].item()), int(v[3][i].item())]
                continue

            # Standard tensors: take the i'th element along batch dimension
            if torch.is_tensor(v):
                if v.ndim >= 1 and v.shape[0] > i:
                    sample[k] = v[i]
                else:
                    sample[k] = v
                continue

            # Anything else: keep as-is
            sample[k] = v

        return sample

    sample0 = _unbatch_sample(batch, 0)

    fig, _axs = plot_sample(sample0, cfg)
    out_path = os.path.join(fig_dir, "dataset_context_test_sample0.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved plot: {out_path}")



    print("\n=== Batch keys ===")
    print(sorted(list(batch.keys())))

    # date may be list[str] depending on collate
    if "date" in batch:
        print("\n=== date ===")
        print(batch["date"])

    if "hr_points" in batch:
        print("\n=== hr_points ===")
        print(batch["hr_points"])
    if "lr_points" in batch:
        print("\n=== lr_points ===")
        print(batch["lr_points"])

    print("\n=== Shapes ===")
    for k in sorted(list(batch.keys())):
        if any(k.endswith(suf) for suf in ("_hr", "_lr")) or k in ("lsm", "topo", "lsm_hr", "sdf"):
            print(f"{k:>20s}: {_shape(batch[k])}")

    # --- Basic shape assertions (spatial dims only)
    hr_size = tuple(cfg["highres"]["data_size"])

    # Expected LR spatial size:
    # - Default: what the UNet sees (lowres.data_size)
    # - Paper2 large_domain: raw LR context size (paper2.spatial_context.lr_context_size)
    paper2_cfg = cfg.get("paper2", {}) or {}
    spatial_cfg = paper2_cfg.get("spatial_context", {}) or {}
    spatial_mode = spatial_cfg.get("mode", None)

    if spatial_mode == "large_domain":
        lr_ctx = spatial_cfg.get("lr_context_size", None)
        if lr_ctx is None:
            raise ValueError("paper2.spatial_context.mode is 'large_domain' but paper2.spatial_context.lr_context_size is missing")
        lr_size = tuple(lr_ctx)
    else:
        lr_size = tuple(cfg["lowres"]["data_size"]) if cfg["lowres"]["data_size"] is not None else hr_size

    # If your pipeline uses resize_factor, mirror the training_utils adjustment
    rf = int(cfg["lowres"].get("resize_factor", 1))
    if rf > 1:
        hr_size = (hr_size[0] // rf, hr_size[1] // rf)
        lr_size = (lr_size[0] // rf, lr_size[1] // rf)

    # Find HR target tensor key
    hr_var = cfg["highres"]["variable"]
    hr_key = f"{hr_var}_hr"
    if hr_key not in batch:
        raise KeyError(f"Expected HR key '{hr_key}' in batch. Found keys: {list(batch.keys())}")

    hr_t = batch[hr_key]
    if not isinstance(hr_t, torch.Tensor):
        raise TypeError(f"{hr_key} is not a torch.Tensor: {type(hr_t)}")

    # Allow shapes like [B, H, W] or [B, C, H, W]
    if hr_t.ndim == 3:
        _, H, W = hr_t.shape
    elif hr_t.ndim == 4:
        _, _, H, W = hr_t.shape
    else:
        raise ValueError(f"Unexpected ndim for {hr_key}: {hr_t.ndim}")

    assert (H, W) == hr_size, f"HR spatial size mismatch: got {(H, W)} expected {hr_size}"

    for cond in cfg["lowres"]["condition_variables"]:
        lr_key = f"{cond}_lr"
        if lr_key not in batch:
            raise KeyError(f"Expected LR key '{lr_key}' in batch.")
        lr_t = batch[lr_key]
        if not isinstance(lr_t, torch.Tensor):
            raise TypeError(f"{lr_key} is not a torch.Tensor: {type(lr_t)}")

        if lr_t.ndim == 3:
            _, h, w = lr_t.shape
        elif lr_t.ndim == 4:
            _, _, h, w = lr_t.shape
        else:
            raise ValueError(f"Unexpected ndim for {lr_key}: {lr_t.ndim}")

        assert (h, w) == lr_size, f"LR spatial size mismatch for {lr_key}: got {(h, w)} expected {lr_size}"

    print(f"\n[paper2][spatial_context] mode={spatial_mode!s} -> expected LR size={lr_size}, expected HR size={hr_size}")
    print("dataset_context_test: basic shape checks passed")


if __name__ == "__main__":
    main()