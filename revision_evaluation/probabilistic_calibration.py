"""Revision diagnostics on finite physical cases: ensemble [member, case], obs [case]."""
import numpy as np
import torch

from sbgm.evaluate.evaluate_prcp.eval_probabilistic.metrics_probabilistic import crps_ensemble
from .common_masks import joint_land_mask
from .thresholds import SEASONS

COVERAGES = (0.50, 0.80, 0.90)
PROBABILITY_EDGES = np.linspace(0, 1, 11)
SUBSETS = ('all_land', 'observed_wet')


def finite_cases(ensemble, observation, land):
    ensemble, observation = np.asarray(ensemble, dtype=float), np.asarray(observation, dtype=float)
    if ensemble.ndim != 3 or ensemble.shape[0] < 2:
        raise ValueError('Expected at least two ensemble members on a spatial grid')
    valid = joint_land_mask(land, observation, ensemble)
    return ensemble[:, valid], observation[valid]


def ratio(total, count):
    return float(total / count) if count else float('nan')


def coverage_rows(ensemble, observation, levels=COVERAGES):
    """Linear quantiles and inclusive endpoints; empty subsets retain zero counts."""
    rows = []
    for level in levels:
        if not 0 < level < 1:
            raise ValueError('Central coverage must lie between zero and one')
        n = observation.size
        if n:
            alpha = (1-level)/2
            lower, upper = np.quantile(ensemble, [alpha, 1-alpha], axis=0, method='linear')
            covered = int(np.count_nonzero((lower <= observation) & (observation <= upper)))
            width_sum = float((upper-lower).sum())
        else:
            covered, width_sum = 0, 0.0
        rows.append(dict(nominal_coverage=level, empirical_coverage=ratio(covered, n),
                         mean_interval_width=ratio(width_sum, n), n_pixels=n,
                         n_covered=covered, interval_width_sum=width_sum))
    return rows


def empirical_crps(ensemble, observation):
    """Reuse the legacy M² empirical kernel in float64, without its empty-mask reduction."""
    if observation.size == 0:
        return np.empty(0)
    ens = torch.from_numpy(np.ascontiguousarray(ensemble[:, None, :], dtype=np.float64))
    obs = torch.from_numpy(np.ascontiguousarray(observation[None, :], dtype=np.float64))
    with torch.no_grad():
        return crps_ensemble(obs, ens, mask=None, reduction='none').numpy().ravel()


def rank_counts(ensemble, observation, rng):
    """Uniform integer rank in [number less, number less + number tied], inclusive."""
    less = (ensemble < observation).sum(axis=0)
    equal = (ensemble == observation).sum(axis=0)
    ranks = less + rng.integers(0, equal+1)
    return np.bincount(ranks, minlength=ensemble.shape[0]+1)


def reliability_counts(ensemble, observation, threshold):
    probabilities = (ensemble >= threshold).mean(axis=0)
    # Internal edges assign [left, right), with p=1 retained in the last bin.
    bins = np.searchsorted(PROBABILITY_EDGES[1:-1], probabilities, side='right')
    rows = []
    for index in range(len(PROBABILITY_EDGES)-1):
        selected = bins == index
        rows.append(dict(threshold=threshold, bin=index,
                         bin_left=float(PROBABILITY_EDGES[index]),
                         bin_right=float(PROBABILITY_EDGES[index+1]),
                         n_cases=int(selected.sum()),
                         forecast_probability_sum=float(probabilities[selected].sum()),
                         n_events=int(np.count_nonzero(observation[selected] >= threshold))))
    return rows


def spread_edges(spreads, n_bins=10):
    """Exact pooled spread quantiles, with repeated edges collapsed; one bin if constant."""
    if n_bins < 1:
        raise ValueError('Spread bin count must be positive')
    values = np.asarray(spreads)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Spread must be finite and nonnegative')
    if values.size == 0:
        return np.array([0.0, 0.0])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins+1), method='linear'))
    return np.repeat(edges, 2) if edges.size == 1 else edges


def spread_skill_counts(spread, squared_error, edges):
    if np.any(spread < edges[0]) or np.any(spread > edges[-1]):
        raise ValueError('Spread outside the pooled first-pass range; inputs may have changed')
    bins = np.searchsorted(edges[1:-1], spread, side='right')
    return [dict(bin=i, bin_left=float(left), bin_right=float(right),
                 n_cases=int(np.count_nonzero(bins == i)),
                 spread_sum=float(spread[bins == i].sum()),
                 squared_error_sum=float(squared_error[bins == i].sum()))
            for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))]


def pool_bins(rows, group_keys, sum_keys, kind):
    """Pool sufficient counts, then form ratios/RMSE (never average daily ratios)."""
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in groups.items():
        result = dict(zip(group_keys, key))
        result.update({k: sum(row[k] for row in group) for k in sum_keys})
        n = result['n_cases']
        result.update(n_dates=len(group), n_valid_dates=sum(row['n_cases'] > 0 for row in group))
        if kind == 'reliability':
            result.update(mean_forecast_probability=ratio(result['forecast_probability_sum'], n),
                          observed_frequency=ratio(result['n_events'], n))
        elif kind == 'spread':
            result.update(mean_spread=ratio(result['spread_sum'], n),
                          rmse=float(np.sqrt(ratio(result['squared_error_sum'], n))))
        else:
            raise ValueError(kind)
        output.append(result)
    return output


def summarize_daily(rows, metric, extra_keys=()):
    """Equal-date means and IQRs; these describe date variability, not confidence intervals."""
    output = []
    keys = sorted({tuple(row[k] for k in ('subset', *extra_keys)) for row in rows})
    for season in SEASONS:
        for key in keys:
            group = [row for row in rows if (season == 'ALL' or row['season'] == season)
                     and tuple(row[k] for k in ('subset', *extra_keys)) == key]
            values = np.array([row[metric] for row in group if np.isfinite(row[metric])])
            result = dict(season=season, **dict(zip(('subset', *extra_keys), key)), metric=metric,
                          mean=float(values.mean()) if values.size else float('nan'),
                          median=float(np.median(values)) if values.size else float('nan'),
                          p25=float(np.quantile(values, .25)) if values.size else float('nan'),
                          p75=float(np.quantile(values, .75)) if values.size else float('nan'),
                          n_dates=len(group), n_valid_dates=int(values.size),
                          n_pixel_days=sum(row['n_pixels'] for row in group))
            output.append(result)
    return output
