"""Strict, shared spatial support. Missing/invalid masks never mean all pixels."""
import numpy as np


def binary_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f'Mask shape {mask.shape} differs from field shape {shape}')
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Land/ROI masks must contain only finite 0/1 values')
    return mask.astype(bool)


def joint_land_mask(land, *fields):
    """For ensembles, require every member finite at the pixel."""
    if not fields or np.asarray(fields[0]).ndim != 2:
        raise ValueError('First field must be a two-dimensional reference')
    shape = np.asarray(fields[0]).shape
    valid = binary_mask(land, shape).copy()
    for field in fields:
        field = np.asarray(field)
        if field.ndim not in (2, 3) or field.shape[-2:] != shape:
            raise ValueError(f'Incompatible field shape: {field.shape}, expected spatial {shape}')
        finite = np.isfinite(field)
        valid &= finite.all(axis=0) if field.ndim == 3 else finite
    return valid


def subset_mask(valid, obs, subset='all_land', wet_threshold=1.0):
    if subset == 'all_land':
        return valid.copy()
    if subset == 'observed_wet':
        return valid & (obs >= wet_threshold)
    raise ValueError(f'Unknown subset: {subset}')
