"""Daily continuous metrics and pooled counts, independent of file I/O and plotting."""
import numpy as np

from sbgm.evaluate.evaluate_prcp.eval_spatial.metrics_spatial import _nanbias, _nanrmse, _nancorr
from .common_masks import joint_land_mask
from .thresholds import WET_THRESHOLD, SEASONS


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def daily_continuous(prediction, observation, mask):
    valid = joint_land_mask(mask, observation, prediction)
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    obs = np.asarray(observation, dtype=np.float64)[valid]
    return dict(bias=_nanbias(pred, obs), mae=float(np.mean(np.abs(pred-obs))) if pred.size else float('nan'),
                rmse=_nanrmse(pred, obs), pearson_r=_nancorr(pred, obs), n_pixels=int(pred.size))


def occurrence_counts(field, mask, threshold=WET_THRESHOLD):
    field = np.asarray(field)
    valid = joint_land_mask(mask, field)
    return dict(n_wet_pixel_days=int(np.count_nonzero(valid & (field >= threshold))),
                n_pixel_days=int(valid.sum()))


def contingency_counts(prediction, observation, mask, threshold):
    valid = joint_land_mask(mask, observation, prediction)
    predicted = np.asarray(prediction)[valid] >= threshold
    observed = np.asarray(observation)[valid] >= threshold
    return dict(hits=int(np.count_nonzero(predicted & observed)),
                misses=int(np.count_nonzero(~predicted & observed)),
                false_alarms=int(np.count_nonzero(predicted & ~observed)),
                correct_negatives=int(np.count_nonzero(~predicted & ~observed)),
                n_pixel_days=int(valid.sum()))


def detection_scores(hits, misses, false_alarms):
    return dict(pod=ratio(hits, hits+misses), far=ratio(false_alarms, hits+false_alarms),
                csi=ratio(hits, hits+misses+false_alarms))


def aggregate_occurrence(daily_rows, methods):
    """ALL and seasonal frequencies from summed wet/valid pixel-day counts."""
    output = []
    for season in SEASONS:
        rows = [r for r in daily_rows if season == 'ALL' or r['season'] == season]
        reference = [r for r in rows if r['method'] == 'danra']
        ref_frequency = ratio(sum(r['n_wet_pixel_days'] for r in reference),
                              sum(r['n_pixel_days'] for r in reference))
        for method in ['danra', *methods]:
            selected = [r for r in rows if r['method'] == method]
            wet = sum(r['n_wet_pixel_days'] for r in selected)
            total = sum(r['n_pixel_days'] for r in selected)
            frequency = ratio(wet, total)
            output.append(dict(method=method, season=season, subset='all_land',
                               wet_threshold=WET_THRESHOLD, wet_frequency=frequency,
                               danra_wet_frequency=ref_frequency, wet_frequency_bias=frequency-ref_frequency,
                               n_wet_pixel_days=wet, n_pixel_days=total, n_dates=len(selected),
                               n_valid_dates=sum(r['n_pixel_days'] > 0 for r in selected)))
    return output


def aggregate_detection(daily_rows, methods, thresholds):
    """Scores of pooled contingency counts, never averages of daily ratios."""
    output = []
    for method in methods:
        for threshold in thresholds:
            rows = [r for r in daily_rows if r['method'] == method and r['threshold'] == threshold]
            counts = {key: sum(r[key] for r in rows) for key in
                      ('hits', 'misses', 'false_alarms', 'correct_negatives', 'n_pixel_days')}
            output.append(dict(method=method, season='ALL', subset='all_land', threshold=threshold,
                               **counts, **detection_scores(counts['hits'], counts['misses'], counts['false_alarms']),
                               n_events=counts['hits']+counts['misses'], n_dates=len(rows),
                               n_valid_dates=sum(r['n_pixel_days'] > 0 for r in rows)))
    return output
