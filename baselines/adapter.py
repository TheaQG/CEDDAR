"""
    Baseline data adapter for DANRA_Dataset_cutouts_ERA5_Zarr
    Assumes each dataset sample includes at least:
        - 'date': str like 'YYYYMMDD'
        - f"{hr_var}_hr": torch.Tensor [1, H, W] (high-res target variable)
        - f"{hr_var}_lr": torch.Tensor [1, H, W] (low-res input variable upsampled to HR grid)
        - "lsm_hr": torch.Tensor [1, H, W] in [0,1] land-sea mask on HR grid
        - Optional extras such as "topo" 

    Adapter returns batched;
        - date: list[str] of length B
        - lr_up: (B, 1, H, W) low-res input upsampled to HR grid
        - y: (B, 1, H, W) high-res target (zeros if absent)
        - lsm: (B, 1, H, W) bool land-sea mask (all True if absent)
        - x_in: (B, C_in, H, W) = concat([lr_up, extra_channels...]) for model input
        - month: (B,) int month 1...12 for each sample
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
from torch.utils.data import DataLoader

def _infer_lr_key(sample: Dict, hr_var: str) -> Optional[str]:
    """ Find the LR-upsample key (prefer f"{hr_var}_lr" if present, else first *_lr) """
    lr_keys = [k for k in sample.keys() if k.endswith("_lr")]
    preferred_key = f"{hr_var}_lr"
    if preferred_key in lr_keys:
        return preferred_key
    return lr_keys[0] if lr_keys else None

@dataclass
class BaselineBatch:
    date: List[str]
    x_in: torch.Tensor   # (B, C_in, H, W)
    y: Optional[torch.Tensor]      # (B, 1, H, W)
    lsm: Optional[torch.Tensor]    # (B, 1, H, W) bool
    lr_up: torch.Tensor  # (B, 1, H, W)
    month: torch.Tensor  # (B,) int month 1...12

class BaselineAdapter:
    def __init__(self, dataset, hr_var: str = "prcp", extra_channels: Optional[List[str]] = None, baseline_type: str = "bilinear"):
        self.dataset = dataset
        self.hr_var = hr_var
        self.extra_channels = extra_channels or []
        self.baseline_type = baseline_type

        probe = self.dataset[0]
        self.lr_key = _infer_lr_key(probe, hr_var=self.hr_var)
        if self.lr_key is None:
            raise KeyError(f"No low-res input key found in dataset sample keys: {list(probe.keys())}. Expected key like '{hr_var}_lr' or '*_lr'.")
        self.has_lsm_hr = "lsm_hr" in probe

    def collate(self, batch: List[Dict]) -> BaselineBatch:
        dates = []
        lr_list, y_list, lsm_list, x_list, months = [], [], [], [], []

        for sample in batch:
            date = str(sample['date'])
            dates.append(date)
            m = int(date[4:6]) if len(date) >= 6 else 1 # month from date
            months.append(m)

            lr_up = sample[self.lr_key]
            assert lr_up.ndim == 3, f"Expected {self.lr_key} to have shape [1,H,W], got {lr_up.shape}"
            lr_list.append(lr_up)

            y = sample.get(f"{self.hr_var}_hr", None)
            if y is not None:
                assert y.ndim == 3, f"Expected {self.hr_var}_hr to have shape [1,H,W], got {y.shape}"
                y_list.append(y)
            else:
                y_list.append(torch.zeros_like(lr_up))  # Placeholder if missing

            if self.has_lsm_hr and sample.get("lsm_hr", None) is not None:
                lsm = sample["lsm_hr"].to(dtype=torch.bool)
            else:
                lsm = torch.ones_like(lr_up, dtype=torch.bool)  # All land if missing
            lsm_list.append(lsm)

            feats = [lr_up]
            for k in self.extra_channels:
                if k in sample and sample[k] is not None:
                    v = sample[k]
                    assert v.shape[-2:] == lr_up.shape[-2:], f"{k} spatial shape {v.shape[-2:]} does not match {self.lr_key} shape {lr_up.shape[-2:]}"
                    feats.append(v)

            x_in = torch.cat(feats, dim=0)  # (C_in, H, W)
            x_list.append(x_in)

        x_in = torch.stack(x_list, dim=0)  # (B, C_in, H, W)
        y = torch.stack(y_list, dim=0) 
        lsm = torch.stack(lsm_list, dim=0) 
        lr_up = torch.stack(lr_list, dim=0)
        month = torch.tensor(months, dtype=torch.int64)

        return BaselineBatch(date=dates, x_in=x_in, y=y, lsm=lsm, lr_up=lr_up, month=month)

    def make_loader(self, batch_size: int = 8, shuffle: bool = False, num_workers: int = 4):
        def _collate_fn(samples):
            return self.collate(samples)
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True, collate_fn=_collate_fn)