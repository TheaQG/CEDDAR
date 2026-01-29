from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Sequence, Union
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaselineDataResolver:
    """
    Baseline resolver that mirrors EvalDataResolver, but without ensembles.

    Expected directory layout (gen_root points at the split level):
        gen_root = .../generated_samples/generation/baselines/<baseline_type>/<split>

    Within gen_root we prefer physical-space folders, with fallbacks to model-space:
        - PMM (or baseline GEN output):
            prefer:  gen_root/pmm_phys/*.npz
            fallback:gen_root/pmm/*.npz
        - HR/LR pairs:
            prefer:  gen_root/lr_hr_phys/*.npz
            fallback:gen_root/lr_hr/*.npz
        - Land-sea masks (optional per-date):
                  gen_root/lsm/*.npz
        - Global land mask (optional, preferred if present):
                  gen_root/meta/land_mask.npz   (key: "lsm_hr" or "mask" or "roi")

    Returns torch.Tensors with shapes consistent with EvalDataResolver:
        HR   → [H, W]
        PMM  → [H, W]     (for baselines, PMM ≡ baseline output)
        LR   → [1, h, w]  (if present)
        MASK → [H, W] bool (if present and eval_land_only=True)
    """

    def __init__(
        self,
        gen_root: str | Path,
        *,
        variable: str = "prcp",
        eval_land_only: bool = True,
        roi_mask_path: Optional[str | Path] = None,
        prefer_phys: bool = True,
        lr_phys_key: Optional[str] = "lr",
    ):
        self.gen_root = Path(gen_root)
        self.variable = variable
        self.eval_land_only = bool(eval_land_only)
        self.prefer_phys = bool(prefer_phys)
        self.lr_phys_key = lr_phys_key

        # Preferred physical-space data paths
        self.dir_pmm_phys  = self.gen_root / "pmm_phys"
        self.dir_lrhr_phys = self.gen_root / "lr_hr_phys"

        # Fallback model-space data paths
        self.dir_pmm_model  = self.gen_root / "pmm"
        self.dir_lrhr_model = self.gen_root / "lr_hr"

        # Masks
        self.dir_lsm = self.gen_root / "lsm"

        # Choose active PMM/HRLR dirs
        self.dir_pmm  = self.dir_pmm_phys  if (self.prefer_phys and self.dir_pmm_phys.exists())  else self.dir_pmm_model
        self.dir_lrhr = self.dir_lrhr_phys if (self.prefer_phys and self.dir_lrhr_phys.exists()) else self.dir_lrhr_model

        if not self.dir_lrhr.exists():
            raise FileNotFoundError(f"[BaselineDataResolver] Missing lr_hr dir: {self.dir_lrhr}")
        if not self.dir_pmm.exists():
            raise FileNotFoundError(f"[BaselineDataResolver] Missing pmm dir: {self.dir_pmm}")

        # === Load global land-sea mask if present (preferred over per-date masks) ===
        self.mask_global: Optional[torch.Tensor] = None
        try:
            p = self.gen_root / "meta" / "land_mask.npz"
            if p.exists():
                arr = np.load(p, allow_pickle=True).get("lsm_hr", None)
                if arr is None:
                    # try common alternates
                    with np.load(p, allow_pickle=True) as d:
                        arr = d.get("mask", None) or d.get("roi", None)
                if arr is not None:
                    m = torch.from_numpy(np.asarray(arr)).to(torch.bool)
                    self.mask_global = self._normalize_hw(m)
                    logger.info(f"[BaselineDataResolver] Loaded global land-sea mask from {p} with shape {self.mask_global.shape}")
        except Exception as e:
            logger.warning(f"[BaselineDataResolver] Failed to load global land-sea mask: {e}")

        # === Load Region-of-Interest mask if provided (intersected later) ===
        self.roi_mask: Optional[torch.Tensor] = None
        if roi_mask_path is not None:
            roi_path = Path(roi_mask_path)
            if roi_path.exists():
                try:
                    arr = np.load(roi_path, allow_pickle=True)
                    if isinstance(arr, np.lib.npyio.NpzFile):  # type: ignore
                        a = arr.get("mask", None) or arr.get("lsm_hr", None) or arr.get("roi", None)
                    else:
                        a = arr
                    if a is not None:
                        m = torch.from_numpy(np.asarray(a)).to(torch.bool)
                        self.roi_mask = self._normalize_hw(m)
                        logger.info(f"[BaselineDataResolver] Loaded ROI mask from {roi_path} with shape {self.roi_mask.shape}")
                except Exception as e:
                    logger.warning(f"[BaselineDataResolver] Failed to load ROI mask from {roi_path}: {e}")

        # Dates are listed from the PMM folder (physical preferred)
        self._dates = sorted([f.stem for f in self.dir_pmm.glob("*.npz")])
        logger.info(f"[BaselineDataResolver] Found {len(self._dates)} dates in PMM folder: {self.dir_pmm}")

    # ------------------------------------------------------------------
    # Small helpers (match EvalDataResolver behavior)
    # ------------------------------------------------------------------
    @staticmethod
    def _npz_to_dict(p: Path) -> Dict[str, np.ndarray]:
        with np.load(p, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}

    @staticmethod
    def _pick_first(d: Dict[str, np.ndarray], candidates: Sequence[str]):
        for k in candidates:
            if k in d:
                return d[k], k
        return None, None

    @staticmethod
    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x))

    @staticmethod
    def _normalize_hw(t: torch.Tensor) -> torch.Tensor:
        # Ensure [H,W] from [1,1,H,W] or [1,H,W]
        if t.dim() == 4 and t.shape[:2] == (1, 1):
            t = t.squeeze(0).squeeze(0)
        elif t.dim() == 3 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t

    @staticmethod
    def _normalize_lr_chw1(t: torch.Tensor) -> torch.Tensor:
        # Ensure [1,h,w] from [h,w], [C,h,w], [1,1,h,w]
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3:
            if t.shape[0] > 1:
                t = t[:1]
        elif t.ndim == 4 and t.shape[:2] == (1, 1):
            t = t.squeeze(0)
        return t

    # ------------------------------------------------------------------
    # Public API (mirrors EvalDataResolver)
    # ------------------------------------------------------------------
    def list_dates(self) -> List[str]:
        return self._dates

    def _load_npz(self, folder: Path, date: str) -> Dict[str, np.ndarray]:
        p = folder / f"{date}.npz"
        if not p.exists():
            raise FileNotFoundError(f"[BaselineDataResolver] Missing NPZ: {p}")
        return self._npz_to_dict(p)

    def load_obs(self, date: str) -> Optional[torch.Tensor]:
        """
        Load HR field for a given date.
        Prefer physical pairs under lr_hr_phys, fall back to lr_hr.
        Keys allowed (first hit wins): 'hr','HR','<var>_hr','<VAR>_HR','obs','OBS'
        Output shape: [H,W] torch.float
        """
        try:
            d = self._load_npz(self.dir_lrhr, date)
        except FileNotFoundError:
            return None
        cand = ("hr", "HR", f"{self.variable}_hr", f"{self.variable.upper()}_HR", "obs", "OBS")
        arr, key = self._pick_first(d, cand)
        if arr is None:
            logger.warning(f"[BaselineDataResolver] HR not found in {self.dir_lrhr}/{date}.npz; tried keys {cand}")
            return None
        t = self._to_tensor(arr)
        return self._normalize_hw(t)

    def load_pmm(self, date: str) -> Optional[torch.Tensor]:
        """
        Load PMM/baseline-generated field for a given date.
        Keys allowed: 'pmm','PMM','gen','GEN', 'qm', 'QM','<var>','<VAR>'
        Output shape: [H,W] torch.float
        """
        try:
            d = self._load_npz(self.dir_pmm, date)
        except FileNotFoundError:
            return None
        cand = ("pmm", "PMM", "gen", "GEN", "qm", "QM", self.variable, self.variable.upper())
        arr, key = self._pick_first(d, cand)
        if arr is None:
            logger.warning(f"[BaselineDataResolver] GEN/PMM not found in {self.dir_pmm}/{date}.npz; tried keys {cand}")
            return None
        t = self._to_tensor(arr)
        return self._normalize_hw(t)

    # For consistency with callers that expect a "generated" field
    def load_gen(self, date: str) -> Optional[torch.Tensor]:
        return self.load_pmm(date)

    def load_ens(self, date: str):
        # Baselines do not have ensembles
        return None

    def has_ensemble(self) -> bool:
        return False

    def load_lr(self, date: str) -> Optional[torch.Tensor]:
        """
        Load LR (native LR grid) for a given date.
        Preferred order: user-requested lr_phys_key, then canonical/common keys, then channel-explicit, then upper-case.
        Output shape: [1,h,w] or None.
        """
        p = self.dir_lrhr / f"{date}.npz"
        if not p.exists():
            return None
        try:
            d = self._npz_to_dict(p)
        except Exception as e:
            logger.warning(f"[BaselineDataResolver] Failed to read {p}: {e}")
            return None

        preferred: List[str] = []
        if isinstance(self.lr_phys_key, str) and len(self.lr_phys_key) > 0:
            preferred.append(self.lr_phys_key)
        preferred.extend([
            # canonical / common
            "lr", "lr_hr", "lr_lrspace", "lr_hrspace",
            # channel-explicit dumps from UNet/QM
            "lr0", "lr1", "lr_up", "lr_native", "lr_phys",
            # upper-case fallbacks
            "LR", "LR_HR", "LR_LRSPACE", "LR_HRSPACE", "LR0", "LR1"
        ])

        arr, chosen = self._pick_first(d, preferred)
        if arr is None:
            logger.info(f"[BaselineDataResolver] No LR arrays found in {p} for any of keys {preferred}")
            return None
        if chosen is not None and chosen != self.lr_phys_key:
            logger.info(f"[BaselineDataResolver] Requested LR key '{self.lr_phys_key}' not found; using '{chosen}' for {date}")

        t = self._to_tensor(arr)
        return self._normalize_lr_chw1(t)

    def load_mask(self, date: str) -> Optional[torch.Tensor]:
        """
        Prefer global land-sea mask over per-date masks under /lsm; intersect with ROI if present.
        Always normalize to [H,W]. Return None if eval_land_only=False.
        """
        if not self.eval_land_only:
            return None

        # Prefer global
        if self.mask_global is not None:
            m = self.mask_global.clone()
        else:
            p = self.dir_lsm / f"{date}.npz"
            if not p.exists():
                return None
            try:
                with np.load(p, allow_pickle=True) as d:
                    arr = d.get("lsm_hr", None) or d.get("mask", None) or d.get("roi", None)
                    if arr is None:
                        return None
                    m = torch.from_numpy(np.asarray(arr)).to(torch.bool)
                    m = self._normalize_hw(m)
            except Exception as e:
                logger.warning(f"[BaselineDataResolver] Failed to load per-date mask from {p}: {e}")
                return None

        # Intersect with ROI if provided and shape-compatible
        if self.roi_mask is not None:
            rm = self.roi_mask
            if rm.shape != m.shape:
                logger.warning(f"[BaselineDataResolver] ROI mask {rm.shape} != LSM {m.shape}; skipping intersection.")
            else:
                m = m & rm
        return m

    # Convenience, mirrors EvalDataResolver
    def fetch(self, date: str) -> Dict[str, Union[torch.Tensor, str, None]]:
        return dict(
            date=date,
            hr=self.load_obs(date),
            pmm=self.load_pmm(date),
            ens=None,
            lr=self.load_lr(date),
            mask=self.load_mask(date),
        )