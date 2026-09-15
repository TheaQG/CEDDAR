"""Inference-only sigma* controls. No model weights or training noise are changed."""
import math

import torch


CONTROL_DEFAULTS = dict(
    sigma_star=1.0, sigma_star_mode="global",
    ramp_start_frac=0.60, ramp_end_frac=0.85,
    ramp_start_sigma=None, ramp_end_sigma=None,
    sigma_star_initial_state="schedule",
)


def sigma_star_kwargs(edm, overrides=None):
    """Use edm settings; the sigma* sweep may explicitly override them."""
    overrides = overrides or {}
    return {key: overrides.get(key, edm.get(key, default))
            for key, default in CONTROL_DEFAULTS.items()}


def build_edm_schedule(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0,
                       S_churn=0.0, S_min=0.0, S_max=float("inf"), S_noise=1.0,
                       sigma_star=1.0, sigma_star_mode="global",
                       ramp_start_frac=0.60, ramp_end_frac=0.85,
                       ramp_start_sigma=None, ramp_end_sigma=None,
                       sigma_star_initial_state="schedule", device="cpu"):
    """Return the actual float32 schedule and its deterministic provenance.

    Global scaling also scales the churn window. Late-ramp scaling leaves that
    window unchanged. The terminal zero is appended after scaling. Invalid or
    increasing trajectories fail before any network calls or random draws.
    """
    if int(num_steps) != num_steps or num_steps < 2:
        raise ValueError("EDM requires num_steps >= 2")
    if not all(math.isfinite(v) and v > 0 for v in (sigma_min, sigma_max, rho, sigma_star)):
        raise ValueError("sigma_min, sigma_max, rho and sigma_star must be finite and positive")
    if sigma_min >= sigma_max:
        raise ValueError("sigma_min must be smaller than sigma_max")
    if (not all(math.isfinite(v) and v >= 0 for v in (S_churn, S_min, S_noise))
            or math.isnan(S_max) or S_max < S_min):
        raise ValueError("Invalid churn parameters or churn window")
    mode = str(sigma_star_mode).lower()
    if mode not in ("global", "late_ramp"):
        raise ValueError("sigma_star_mode must be global or late_ramp")
    if sigma_star_initial_state not in ("schedule", "legacy_sigma_max"):
        raise ValueError("sigma_star_initial_state must be schedule or legacy_sigma_max")

    idx = torch.arange(num_steps, device=device, dtype=torch.float32)
    u = idx / (num_steps - 1)
    base = (sigma_max ** (1 / rho) + u * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    i0 = i1 = None
    if mode == "global":
        factors = torch.full_like(base, float(sigma_star))
    else:
        if (ramp_start_sigma is None) != (ramp_end_sigma is None):
            raise ValueError("Supply both ramp sigma thresholds, or neither")
        if ramp_start_sigma is not None:
            if not (sigma_min <= ramp_end_sigma < ramp_start_sigma <= sigma_max):
                raise ValueError("Ramp thresholds must satisfy sigma_min <= end < start <= sigma_max")
            # Thresholds refer to the unmodified schedule, as in v1.0.2.
            indices = [(base <= threshold).nonzero() for threshold in (ramp_start_sigma, ramp_end_sigma)]
            if any(len(i) == 0 for i in indices):
                raise ValueError("Ramp threshold is not reached by the numerical schedule")
            i0, i1 = (int(i[0].item()) for i in indices)
        else:
            if not (0 <= ramp_start_frac < ramp_end_frac <= 1):
                raise ValueError("Ramp fractions must satisfy 0 <= start < end <= 1")
            i0 = round(ramp_start_frac * (num_steps - 1))
            i1 = round(ramp_end_frac * (num_steps - 1))
        if i0 >= i1:
            raise ValueError("Ramp needs distinct start/end steps; increase steps or widen the ramp")
        t = ((idx - i0) / (i1 - i0)).clamp(0, 1)
        weights = t * t * (3 - 2 * t)
        factors = 1 + (float(sigma_star) - 1) * weights

    positive = factors * base
    if not (torch.isfinite(positive).all() and (positive > 0).all()
            and (positive[1:] < positive[:-1]).all()):
        raise ValueError("sigma* must produce a finite, positive, strictly decreasing schedule")
    sigmas = torch.cat([positive, positive.new_zeros(1)])
    lower, upper = ((S_min * sigma_star, S_max * sigma_star) if mode == "global" else (S_min, S_max))
    # Use the analytic first endpoint: preserves the v1.0.2 alpha=1 baseline
    # exactly, avoiding a roundoff-only change from sigma_max to base[0].
    initial_std = float(sigma_max) * (float(factors[0]) if sigma_star_initial_state == "schedule" else 1.0)
    gamma = torch.tensor([min(S_churn / num_steps, math.sqrt(2) - 1)
                          if lower <= float(s) <= upper and S_churn > 0 else 0.0
                          for s in positive], device=device, dtype=positive.dtype)
    sigma_hat = positive * (1 + gamma)
    details = dict(
        implementation="paper1-revision-sigma-v1", base_sigmas=base.tolist(),
        sigmas=sigmas.tolist(), scale_factors=factors.tolist(),
        ramp_start_index=i0, ramp_end_index=i1, initial_std=initial_std,
        S_min_effective=lower, S_max_effective=upper, gamma=gamma.tolist(),
        churn_noise_std=(S_noise * (sigma_hat.square() - positive.square()).sqrt()).tolist(),
        sigma_hat=sigma_hat.tolist(), step_deltas=(sigmas[1:] - sigma_hat).tolist(),
    )
    return sigmas, details
