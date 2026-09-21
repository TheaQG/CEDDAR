"""Wernli et al. (2008) SAL components with documented fixed physical object thresholds.

The legacy structure proxies are deliberately not reused. Distances use equal-spacing
grid coordinates; their common physical pixel size cancels in normalization by d.
Reference: https://doi.org/10.1175/2008MWR2415.1, equations 2, 4–9.
"""
import numpy as np
from scipy.ndimage import maximum
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from .object_metrics import label_objects


def domain_diameter(valid):
    """Largest separation of valid grid-cell centres; hull avoids an N×N distance array."""
    points = np.column_stack(np.nonzero(valid))
    if len(points) < 2:
        return float('nan')
    try:
        boundary = points[ConvexHull(points).vertices]
    except QhullError:  # A line or two pixels: lexicographic extremes are the endpoints.
        boundary = points[[0, -1]]
    return float(pdist(boundary).max())


def field_sal(field, valid, threshold):
    """Domain mean/centroid, mass-weighted scaled volume and object-centroid scatter."""
    values = field[valid]
    labels, count = label_objects(valid & (field >= threshold))
    result = dict(mean=float('nan'), centre=np.array([np.nan, np.nan]),
                  volume=float('nan'), scatter=float('nan'), n_objects=int(count),
                  n_negative_pixels=int(np.count_nonzero(values < 0)))
    if not values.size or result['n_negative_pixels']:
        return result  # SAL assumes nonnegative precipitation; never silently clip.
    result['mean'] = float(values.mean())
    total = values.sum()
    if total <= 0:
        return result
    yy, xx = np.indices(field.shape)
    centre = np.array([(yy[valid]*values).sum(), (xx[valid]*values).sum()])/total
    result['centre'] = centre
    if not count:
        return result
    selected = labels > 0
    ids, weights = labels[selected], field[selected]
    mass = np.bincount(ids, weights=weights, minlength=count+1)[1:]
    peak = maximum(field, labels=labels, index=np.arange(1, count+1))
    centres = np.column_stack([np.bincount(ids, weights=axis[selected]*weights,
                                          minlength=count+1)[1:]/mass for axis in (yy, xx)])
    # V_n = mass_n / peak_n; V = sum(mass_n V_n) / sum(mass_n).
    result['volume'] = float(np.sum(mass*(mass/peak))/mass.sum())
    result['scatter'] = float(np.sum(mass*np.linalg.norm(centres-centre, axis=1))/mass.sum())
    return result


def sal_components(reference, prediction, diameter):
    """Component-specific NaNs: A can exist without threshold-exceeding objects."""
    def difference(a, b):
        return float(2*(a-b)/(a+b)) if np.isfinite(a+b) and a+b > 0 else float('nan')

    a = difference(prediction['mean'], reference['mean'])
    s = difference(prediction['volume'], reference['volume'])
    l1 = l2 = float('nan')
    if np.isfinite(diameter) and diameter > 0:
        l1 = float(np.linalg.norm(prediction['centre']-reference['centre'])/diameter)
        l2 = float(2*abs(prediction['scatter']-reference['scatter'])/diameter)
    return dict(S=s, A=a, L=l1+l2, L1=l1, L2=l2,
                n_objects_reference=reference['n_objects'], n_objects_prediction=prediction['n_objects'],
                n_negative_reference=reference['n_negative_pixels'],
                n_negative_prediction=prediction['n_negative_pixels'])
