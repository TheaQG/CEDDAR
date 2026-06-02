
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional

import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _savefig(fig, out_path: Path, dpi: int = 300):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(out_path, dpi=dpi)
    logger.debug(f"[plot_utils] Saved figure → {out_path}")
    plt.close(fig)


def _nice():
    # lightweight styling; only apply once to avoid repeated global overrides
    if getattr(_nice, "_applied", False):
        return
    plt.rcParams.update({
        "figure.figsize": (5.5, 4.0),
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })
    _nice._applied = True

def _to_date_safe(s: str) -> Optional[datetime]:
    s_clean = s.strip()
    try:
        if len(s_clean) == 8 and s_clean.isdigit():
            return datetime.strptime(s_clean, "%Y%m%d")
        return datetime.fromisoformat(s_clean)
    except Exception:
        logger.debug(f"[plot_utils] Failed to parse date: {s}")
        return None


def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"

