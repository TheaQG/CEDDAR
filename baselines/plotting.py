# baselines/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scor_dm.plotting_utils import get_cmap_for_variable

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _mask_sea(arr: np.ndarray, lsm: np.ndarray | None):
    if lsm is None:
        return arr
    # lsm is expected [1,H,W] or [H,W] in {0,1}; mask out sea by setting to nan
    m = lsm[0] if lsm.ndim == 3 else lsm
    out = arr.copy()
    out[..., m == 0] = np.nan
    return out

def _squeeze2d(x: np.ndarray | None):
    if x is None:
        return None
    x = np.asarray(x)
    # squeeze singleton axes repeatedly
    while x.ndim > 2 and 1 in x.shape:
        x = np.squeeze(x)
    # If still 3D (e.g., [C,H,W]), pick the first channel
    if x.ndim == 3 and x.shape[0] <= 4:
        x = x[0]
    return x

def plot_triplet(hr_var: str,               # for setting default cmap
                 date: str,
                 pmm: np.ndarray,          # [1,1,H,W] or [1,H,W] or [H,W]
                 hr: np.ndarray | None,    # same layout; may be None on test-only
                 lr: np.ndarray | None,    # same layout; may be None
                 lsm: np.ndarray | None,
                 out_dir: Path,
                 vmax: float | None = None,
                 title_suffix: str = "") -> Path:
    """
    Save a PNG with up to three panels: prediction (pmm), HR truth (if available), LR upsample (if available).
    """
    _ensure_dir(out_dir)

    p = _squeeze2d(pmm)
    t = _squeeze2d(hr)
    l = _squeeze2d(lr)
    m = _squeeze2d(lsm)

    p = _mask_sea(p, m) if p is not None else None
    t = _mask_sea(t, m) if t is not None else None
    l = _mask_sea(l, m) if l is not None else None

    cmap = get_cmap_for_variable(hr_var) if hr_var else "Blues"

    # choose vmax: either provided, or common 99th percentile across available maps
    if vmax is None:
        vals = []
        for a in (p, t, l):
            if a is not None and np.isfinite(a).any():
                vals.append(np.nanpercentile(a, 99))
        vmax = float(np.max(vals)) if vals else None

    panels = [("Prediction", p)]
    if t is not None:
        panels.append(("HR truth", t))
    if l is not None:
        panels.append(("LR↑", l))

    n = len(panels)
    fig = plt.figure(figsize=(4*n, 4), dpi=120)
    im = None
    for i, (name, img) in enumerate(panels, 1):
        ax = fig.add_subplot(1, n, i)
        if img is not None:
            if vmax is not None:
                im = ax.imshow(img, cmap=cmap, vmin=0.0, vmax=vmax)
            else:
                im = ax.imshow(img, cmap=cmap, vmin=0.0)
        else:
            # Show a blank panel if img is None
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_facecolor("lightgray")
        ax.set_title(f"{name}")
        ax.axis("off")
        ax.invert_yaxis()
    if im is not None:
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cax, label="mm/day")
    fig.suptitle(f"{date} {title_suffix}", y=0.98)
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))

    out_path = out_dir / f"{date}.png"
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return out_path

def plotting_enabled(cfg) -> bool:
    return bool(cfg.get("baseline", {}).get("plotting", {}).get("enabled", False))

def plotting_params(cfg):
    p = cfg.get("baseline", {}).get("plotting", {})
    return dict(
        max_plots = int(p.get("max_per_split", 16)),
        cmap      = str(p.get("cmap", "Blues")),
        vmax      = p.get("vmax", None),
        subdir    = str(p.get("subdir", "samples/baselines")),
    )

def resolve_samples_dir(cfg, baseline_type: str, split: str) -> Path:
    """
    target: SAMPLE_DIR = ROOT_DIR/models_and_samples/generated_samples + '/samples/baselines/{baseline_type}/{split}'
    """
    # prefer 'sample_dir' if present, else 'samples_dir'
    root_key = "sample_dir" if "sample_dir" in cfg["paths"] else "samples_dir"
    root = Path(cfg["paths"][root_key])
    base = root / "samples" / "baselines" / baseline_type / split
    _ensure_dir(base)
    return base
