"""Land-only connected objects, in pixels; no smoothing or minimum-size filtering."""
import numpy as np
from scipy.ndimage import label

THRESHOLDS = (1., 5., 10., 20.)


def label_objects(mask):
    """Diagonal neighbours belong to the same object (8-neighbour connectivity)."""
    return label(mask, structure=np.ones((3, 3), dtype=int))


def object_metrics(field, valid, exceedance):
    selected = np.asarray(exceedance, dtype=bool) & valid
    labels, count = label_objects(selected)
    areas = np.bincount(labels.ravel())[1:]
    n, wet = int(np.count_nonzero(valid)), int(selected.sum())
    values = field[selected]
    return dict(n_pixels=n, n_objects=int(count) if n else float('nan'),
                wet_pixel_count=wet, wet_fraction=wet/n if n else float('nan'),
                mean_object_area=float(areas.mean()) if count else float('nan'),
                median_object_area=float(np.median(areas)) if count else float('nan'),
                largest_object_fraction=float(areas.max()/wet) if count else float('nan'),
                mean_exceedance_intensity=float(values.mean()) if wet else float('nan'),
                max_exceedance_intensity=float(values.max()) if wet else float('nan'))


def equal_area_mask(field, reference, valid, threshold):
    """Linear quantile with inclusive ties. A zero target explicitly selects no pixels."""
    n = int(valid.sum())
    target_count = int(np.count_nonzero(valid & (reference >= threshold)))
    fraction = target_count/n if n else float('nan')
    if target_count:
        effective = float(np.quantile(field[valid], 1-fraction, method='linear'))
        selected = valid & (field >= effective)
    else:
        # Q_1 with >= would wrongly retain the maximum when DANRA has no event.
        effective, selected = float('nan'), np.zeros_like(valid, dtype=bool)
    achieved = float(selected.sum()/n) if n else float('nan')
    return selected, dict(effective_threshold=effective, target_wet_fraction=fraction,
                         achieved_wet_fraction=achieved, wet_fraction_difference=achieved-fraction,
                         target_wet_pixel_count=target_count)
