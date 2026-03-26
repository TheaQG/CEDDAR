from __future__ import annotations
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    date: str
    hr: Optional[torch.Tensor]        # [H,W]
    pmm: Optional[torch.Tensor]       # [H,W]
    ens: Optional[torch.Tensor]       # [M,H,W]
    lr: Optional[torch.Tensor]        # [1,h,w]
    mask: Optional[torch.Tensor]      # [H,W] bool


class EvalDataResolver:
    """Centralized access to already generated data for evaluation."""

    def __init__(
        self,
        gen_root: str | Path,
        eval_land_only: bool = True,
        roi_mask_path: Optional[str | Path] = None,
        prefer_phys: bool = True,
        lr_phys_key: Optional[str] = "lr",
    ):
        self.gen_root = Path(gen_root)
        self.eval_land_only = eval_land_only
        self.roi_mask_path = Path(roi_mask_path) if roi_mask_path is not None else None
        self.prefer_phys = prefer_phys
        self.lr_phys_key = lr_phys_key

        # Preferred physical-space data paths
        self.dir_ens_phys = self.gen_root / "ensembles_phys"
        self.dir_pmm_phys = self.gen_root / "pmm_phys"
        self.dir_lrhr_phys = self.gen_root / "lr_hr_phys"

        # Fallback model-space data paths
        self.dir_ens_model = self.gen_root / "ensembles"
        self.dir_pmm_model = self.gen_root / "pmm"
        self.dir_lrhr_model = self.gen_root / "lr_hr"

        # Masks
        self.dir_lsm = self.gen_root / "lsm"

        # === Load global land-sea mask if present ===
        self.mask_global: Optional[torch.Tensor] = None
        try:
            p = self.gen_root / "meta" / "land_mask.npz"
            if p.exists():
                arr = np.load(p, allow_pickle=True).get("lsm_hr", None)
                if arr is not None:
                    self.mask_global = torch.from_numpy(np.asarray(arr)).to(torch.bool)
                    logger.info(
                        f"[EvalDataResolver] Loaded global land-sea mask from {p} with shape {self.mask_global.shape}"
                    )
        except Exception as e:
            p = self.gen_root / "meta" / "land_mask.npz"
            logger.warning(f"[EvalDataResolver] Failed to load global land-sea mask from {p}: {e}")

        # === Load Region-of-Interest mask if provided ===
        self.roi_mask: Optional[torch.Tensor] = None
        if roi_mask_path is not None:
            roi_path = Path(roi_mask_path)
            if roi_path.exists():
                try:
                    arr = np.load(roi_path, allow_pickle=True)
                    if isinstance(arr, np.lib.npyio.NpzFile):  # type: ignore
                        a = None
                        for k in ("mask", "lsm_hr", "roi"):
                            if k in arr.files:
                                a = arr[k]
                                break
                    else:
                        a = arr
                    if a is not None:
                        m = torch.from_numpy(np.asarray(a)).to(torch.bool)
                        # normalize to [H,W]
                        if m.dim() == 4 and m.shape[:2] == (1, 1):
                            m = m.squeeze(0).squeeze(0)
                        elif m.dim() == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        self.roi_mask = m
                        logger.info(
                            f"[EvalDataResolver] Loaded ROI mask from {roi_path} with shape {self.roi_mask.shape}"
                        )
                except Exception as e:
                    logger.warning(f"[EvalDataResolver] Failed to load ROI mask from {roi_path}: {e}")

    # =====
    # Basic helpers
    # =====
    def list_dates(self) -> list[str]:
        """Return sorted date stems from PMM folder (physical preferred)."""
        base = self.dir_pmm_phys if self.dir_pmm_phys.exists() and self.prefer_phys else self.dir_pmm_model
        dates = sorted([f.stem for f in base.glob("*.npz")])
        logger.info(f"[EvalDataResolver] Found {len(dates)} dates in PMM folder: {base}")
        return dates

    def _load_npz(self, folder: Path, date: str, key: str):
        p = folder / f"{date}.npz"
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        return d.get(key, None)

    def _subsample_members(self, ens: torch.Tensor, n: Optional[int], seed: int = 1234) -> torch.Tensor:
        """Optionally subsample ensemble members along dim 0 to size n (without replacement)."""
        if ens is None or n is None:
            return ens
        m = ens.shape[0]
        if n >= m:
            return ens
        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(m, generator=g)[:n]
        return ens.index_select(0, idx)

    # =====
    # Data loaders (HR, PMM, ensembles, LR, mask)
    # =====
    def load_obs(self, date: str) -> Optional[torch.Tensor]:
        """Load HR field for a given date. Output: [H,W]."""
        x = self._load_npz(self.dir_lrhr_phys, date, "hr")
        if x is None:
            x = self._load_npz(self.gen_root / "lr_hr", date, "hr")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(0)
        return t.squeeze(0)

    def load_pmm(self, date: str) -> Optional[torch.Tensor]:
        """Load PMM field for a given date. Output: [H,W]."""
        x = self._load_npz(self.dir_pmm_phys, date, "pmm")
        if x is None:
            x = self._load_npz(self.dir_pmm_model, date, "pmm")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(0)
        return t.squeeze(0)

    def load_ens(self, date: str) -> Optional[torch.Tensor]:
        """Load ensemble for a given date. Output: [M,H,W]."""
        x = self._load_npz(self.dir_ens_phys, date, "ens")
        if x is None:
            x = self._load_npz(self.dir_ens_model, date, "ens")
        if x is None:
            return None
        t = torch.from_numpy(np.asarray(x)).squeeze(1)  # [M,1,H,W] → [M,H,W]
        return t

    def load_lr(self, date: str) -> Optional[torch.Tensor]:
        """Load LR input for a given date. Prefer physical-space LR if available and configured. Output: [1,h,w] or None."""
        p = self.dir_lrhr_phys / f"{date}.npz"
        if not p.exists():
            return None
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            logger.warning(f"[EvalDataResolver] Failed to read {p}: {e}")
            return None

        x = None
        chosen = None
        desired = (self.lr_phys_key if isinstance(self.lr_phys_key, str) and len(self.lr_phys_key) > 0 else None)

        if desired is not None and desired in d.files:
            x = d[desired]
            chosen = desired
        elif desired is not None:
            logger.info(
                f"[EvalDataResolver] Requested LR key '{desired}' not found in {p}; returning None to avoid wrong channel."
            )
            return None
        else:
            for k in ("lr", "lr_lrspace"):
                if k in d.files and d[k] is not None:
                    x = d[k]
                    chosen = k
                    break

        if x is None:
            logger.info(
                f"[EvalDataResolver] No LR arrays found in {p} for any of keys ['lr', 'lr_lrspace']; returning None to avoid wrong channel."
            )
            return None

        if chosen is not None and chosen != self.lr_phys_key:
            logger.info(
                f"[EvalDataResolver] Requested LR key '{self.lr_phys_key}' not found; using '{chosen}' instead for {date}"
            )

        x_arr = np.asarray(x)
        if x_arr.dtype == np.object_:
            if x_arr.size == 1 and x_arr.item() is None:
                logger.warning(f"[EvalDataResolver] LR key '{chosen}' is None in {p}; returning None")
                return None
            logger.warning(
                f"[EvalDataResolver] LR key '{chosen}' has dtype=object in {p} (shape={x_arr.shape}); returning None"
            )
            return None

        lr_t = torch.from_numpy(x_arr)
        if lr_t.ndim == 3:
            lr_t = lr_t[0:1, ...]
        elif lr_t.ndim == 2:
            lr_t = lr_t.unsqueeze(0)
        return lr_t
    
    def load_mask(self, date: str) -> Optional[torch.Tensor]:
        """Load land-sea mask for a date, normalized to [H,W]."""
        if not self.eval_land_only:
            return None

        m: Optional[torch.Tensor] = None

        # 1) Prefer per-date mask
        p_date = self.dir_lsm / f"{date}.npz"
        if p_date.exists():
            try:
                arr = np.load(p_date, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):  # type: ignore
                    a = None
                    for k in ("lsm_hr", "lsm", "mask"):
                        if k in arr.files:
                            a = arr[k]
                            break
                else:
                    a = arr
                if a is not None:
                    m = torch.from_numpy(np.asarray(a)).to(torch.bool)
                    logger.info(
                        f"[EvalDataResolver] Loaded per-date land-sea mask from {p_date} with shape {tuple(m.shape)}"
                    )
            except Exception as e:
                logger.warning(f"[EvalDataResolver] Failed to load per-date land-sea mask from {p_date}: {e}")

        # 2) Fall back to global mask
        if m is None:
            if self.mask_global is not None:
                m = self.mask_global.clone()
            else:
                p = self.dir_lsm / f"{date}.npz"
                if not p.exists():
                    return None
                try:
                    arr = np.load(p, allow_pickle=True).get("lsm_hr", None)
                    if arr is None:
                        return None
                    m = torch.from_numpy(np.asarray(arr)).to(torch.bool)
                except Exception as e:
                    logger.warning(f"[EvalDataResolver] Failed to load land-sea mask from {p}: {e}")
                    return None

        # 3) Normalize to [H,W]
        if m.dim() == 4 and m.shape[:2] == (1, 1):
            m = m.squeeze(0).squeeze(0)
        elif m.dim() == 3 and m.shape[0] == 1:
            m = m.squeeze(0)

        # 4) Intersect with ROI mask
        if self.roi_mask is not None:
            rm = self.roi_mask
            if rm.shape != m.shape:
                if rm.dim() == 4 and rm.shape[:2] == (1, 1):
                    rm = rm.squeeze(0).squeeze(0)
                elif rm.dim() == 3 and rm.shape[0] == 1:
                    rm = rm.squeeze(0)
            if rm.shape == m.shape:
                m = m & rm

        return m

    def fetch(
        self,
        date: str,
        want_ensemble: bool = True,
        n_members: Optional[int] = None,
        seed: int = 1234,
    ) -> EvalSample:
        """Unified access to all evaluation arrays for a given date."""
        hr = self.load_obs(date)
        pmm = self.load_pmm(date)
        ens = self.load_ens(date) if want_ensemble else None
        if ens is not None:
            ens = self._subsample_members(ens, n_members, seed)
        lr = self.load_lr(date)
        mask = self.load_mask(date)
        return EvalSample(date=date, hr=hr, pmm=pmm, ens=ens, lr=lr, mask=mask)