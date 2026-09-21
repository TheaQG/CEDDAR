"""
Dry-bias decomposition helpers

Primary diagnostic separates precipitation occurrence from intensity:
    mean precipitation = wet contribution + below-threshold contribution
where 
    wet contribution = P(P >= q) * E[P | P >= q]

Each deterministic method is conditioned on its own pixels.

"""
import numpy as np

from .thresholds import WET_THRESHOLD

QUANTILES = (0.50, 0.90, 0.99)

COUNT_KEYS = (
    "n_pixel_days",
    "n_wet_pixel_days",
    "n_dry_pixel_days",
    "precip_sum",
    "wet_precip_sum",
    "dry_precip_sum",
)


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float("nan")


def empty_counts():
    return dict(
        n_pixel_days=0,
        n_wet_pixel_days=0,
        n_dry_pixel_days=0,
        precip_sum=0.0,
        wet_precip_sum=0.0,
        dry_precip_sum=0.0,
    )


def field_components(field, valid, threshold=WET_THRESHOLD):
    """Return sufficient pooled counts plus this method's own wet values."""
    field = np.asarray(field, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)

    if field.shape != valid.shape:
        raise ValueError(
            f"Field shape {field.shape} differs from mask shape {valid.shape}"
        )

    values = field[valid]

    if not np.isfinite(values).all():
        raise ValueError("Valid mask includes non-finite precipitation values")

    wet = values >= threshold
    wet_values = values[wet]
    dry_values = values[~wet]

    counts = dict(
        n_pixel_days=int(values.size),
        n_wet_pixel_days=int(wet_values.size),
        n_dry_pixel_days=int(dry_values.size),
        precip_sum=float(values.sum()),
        wet_precip_sum=float(wet_values.sum()),
        dry_precip_sum=float(dry_values.sum()),
    )

    return counts, wet_values


def add_counts(target, source):
    """Accumulate one date into an existing pooled group."""
    for key in COUNT_KEYS:
        target[key] += source[key]


def summarize_components(counts):
    """Convert pooled counts into occurrence/intensity diagnostics"""
    n = counts["n_pixel_days"]
    n_wet = counts["n_wet_pixel_days"]
    n_dry = counts["n_dry_pixel_days"]

    # How often wet and dry conditions occur
    wet_frequency = ratio(n_wet, n)
    dry_frequency = ratio(n_dry, n)

    # How much precipitation occurs when wet or dry on average
    conditional_mean_wet = ratio(counts["wet_precip_sum"], n_wet)
    conditional_mean_dry = ratio(counts["dry_precip_sum"], n_dry)

    # Overall mean precipitation
    mean_precip = ratio(counts["precip_sum"], n)

    # Contributions to the unconditional mean, how much wet and dry conditions contribute
    wet_contribution = ratio(counts["wet_precip_sum"], n)
    dry_contribution = ratio(counts["dry_precip_sum"], n)

    # How well the contributions reconstruct the overall mean
    reconstructed = (
        wet_contribution + dry_contribution if n else float("nan")
    )

    return dict(
        **counts,
        wet_frequency=wet_frequency,
        dry_frequency=dry_frequency,
        mean_precip=mean_precip,
        conditional_mean_wet=conditional_mean_wet,
        conditional_mean_dry=conditional_mean_dry,
        wet_contribution=wet_contribution,
        dry_contribution=dry_contribution,
        reconstruction_error=(mean_precip - reconstructed) if n else float("nan"),
    )


def summarize_wet_values(chunks, quantiles=QUANTILES):
    """Conditional quantiles pooled over method-specific wet pixel-days"""
    nonempty = [
        np.asarray(chunk, dtype=np.float64) for chunk in chunks if np.asarray(chunk).size
    ]

    q_names = [f"p{int(q*100)}" for q in quantiles]

    if not nonempty:
        return dict(
            n_wet_pixel_days=0,
            conditional_mean_wet=float("nan"),
            **{q_name: float("nan") for q_name in q_names},
        )

    values = np.concatenate(nonempty)

    qs = np.quantile(values, quantiles, method="linear")

    return dict(
        n_wet_pixel_days=values.size,
        conditional_mean_wet=float(values.mean()),
        **{q_name: q for q_name, q in zip(q_names, qs)},
    )
    