"""Read saved legacy artifacts, without rerunning evaluation or inverse transforms.

Evaluation loaders return {tables, arrays, metadata, sources, directory, origin}.
Keys are file stems; CSV tables are lists of string-valued rows (empty = missing),
as in revision_evaluation.plot_common. NPZ contents are dictionaries of arrays.
No rows, dates, members or NaNs are silently removed. Paths may be overridden.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..paths import LEGACY_ROOT

MODEL_NAME = ('B1_GSDF_RGBCE__HR_prcp_DANRA__SIZE_128x128__LR_prcp_ERA5'
              '__LOSS_sdfweighted__HEADS_4__TIMESTEPS_56')

DEFAULT_EVALUATION = (LEGACY_ROOT / 'legacy_evaluation/SBGM_SD/models_and_samples'
                      / 'generated_samples/evaluation' / MODEL_NAME / 'prcp')

GENERATION_ROOT = (LEGACY_ROOT / 'revision_inputs/SBGM_SD/models_and_samples'
                   / 'generated_samples/generation')

DEFAULT_GENERATION = GENERATION_ROOT / MODEL_NAME

DEFAULT_BASELINES = {name: GENERATION_ROOT / 'baselines' / directory / 'test'
                     for name, directory in [('era5_bilinear', 'bilinear'), ('qm', 'qm')]}


def _npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _metadata(result, directory):
    """Retain saved metadata verbatim, including requested versus effective settings."""

    directory = Path(directory)

    for pattern in ('*.json', '*.yaml', '*.yml', '*.meta.txt', 'meta/*.json', 'meta/*.yaml', 'meta/*.yml'):
        for path in sorted(directory.glob(pattern)):
            key = str(path.resolve())

            if key in result['metadata']:
                continue

            text = path.read_text()
            value = json.loads(text) if path.suffix == '.json' else (
                yaml.safe_load(text) if path.suffix in ('.yaml', '.yml') else text)

            result['metadata'][key] = value
            result['sources'][key] = key


def _bundle(directory, required=(), prefix=''):
    """Bundle CSV and NPZ tables from a directory, retaining metadata and sources."""

    directory = Path(directory).expanduser().resolve()

    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Missing {directory / name}; supply the correct evaluation directory')
    result: dict[str, Any] = dict(directory=directory, tables={}, arrays={}, metadata={}, sources={})

    for path in sorted(directory.glob(f'{prefix}*')):
        if path.suffix == '.csv':
            with path.open(newline='') as stream:
                result['tables'][path.stem] = list(csv.DictReader(stream))
        elif path.suffix == '.npz':
            result['arrays'][path.stem] = _npz(path)
        else:
            continue
        result['sources'][path.name] = str(path)

    if not result['tables'] and not result['arrays']:
        raise FileNotFoundError(f'No saved CSV/NPZ data in {directory}')

    _metadata(result, directory)
    return result


def _evaluation(family, directory, required=(), prefix=''):
    """Accept a prcp root, model evaluation root, family root, or tables folder."""

    root = Path(directory if directory is not None else DEFAULT_EVALUATION).expanduser().resolve()
    candidates = {p for p in (root, root/'tables', root/family/'tables', root/'prcp'/family/'tables')
                  if all((p/name).is_file() for name in required) and p.is_dir()}

    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one {family} tables directory below {root}; found {len(candidates)}. '
                                f'Pass its tables directory explicitly. Required files: {required}')

    tables = candidates.pop()
    result = _bundle(tables, required, prefix)
    _metadata(result, tables.parent)

    # Model-level metadata, when present, is useful for date/split provenance.
    if tables.parent.parent.name == 'prcp':
        _metadata(result, tables.parent.parent.parent)

    result['origin'] = 'legacy'
    return result


def load_psd(directory=None):
    """scale_psd_curves: psd_gen is PMM; psd_gen_ens_mean is mean member PSD.

    The latter is NOT PSD of the ensemble mean. Keep original keys and CI arrays.
    Load QM/bilinear evaluation directories separately; no spectral interpolation.
    """
    return _evaluation('scale', directory, ('scale_psd_curves.npz',), 'scale_')


def load_seasonal_distributions(directory=None):
    """Load daily histograms and return season row indices (no PDF renormalization).

    dist_daily.counts_gen describes saved PMM, NOT pooled ensemble members.
    dist_member_histograms / dist_gen_ens_pool are separate ensemble artifacts;
    they do not supply per-season member counts unless explicitly saved as such.
    """
    result = _evaluation('distributional', directory, ('dist_daily.npz',), 'dist_')

    dates = result['arrays']['dist_daily']['dates'].astype(str)

    months = np.array([datetime.strptime(d.replace('-', ''), '%Y%m%d').month for d in dates])

    result['season_indices'] = {season: np.flatnonzero(np.isin(months, members)) for season, members in
                                dict(DJF=(12, 1, 2), MAM=(3, 4, 5), JJA=(6, 7, 8), SON=(9, 10, 11)).items()}
    return result


def load_extremes(directory=None):
    """Tail statistics and saved metadata (wet threshold and pooled-pixel/day basis)."""
    return _evaluation('extremes', directory, ('ext_tails.csv',), 'ext_')


def load_probabilistic(directory=None):
    """Legacy PIT/rank, daily CRPS, reliability and spread-skill, unchanged."""
    return _evaluation('probabilistic', directory, ('prob_crps_daily.csv',), 'prob_')


def load_spatial(directory=None):
    """Saved annual/seasonal maps; never substitute an absent median/QM/PMM map.

    spatial_ensmean_<group> means the member average of the saved spatial statistic;
    sums reflect available dates, not necessarily a complete calendar year.
    """
    return _evaluation('spatial', directory, ('spatial_summary.csv',), 'spatial_')


def load_sigma_star(directory=None):
    """Keep daily metrics, PSD curves and sigma_control_meta.json together.

    Legacy ramp labels can be requested config rather than actual sampler mode.
    Revision files use the same table layout; revision.load_sigma_star adds the
    resolved run config. No SEM/STD or metric aggregation is recalculated here.
    """
    return _evaluation('sigma_control', directory,
                       ('metrics_by_sigma.csv', 'sigma_psd_curves.npz'))


def _field(arrays, keys, path):
    """Extract a single 2D field from a dictionary of arrays."""

    for key in keys:
        if key in arrays:
            array = arrays[key]

            while array.ndim > 2 and array.shape[0] == 1:
                array = array[0]
            if array.ndim != 2:
                raise ValueError(f'{path}: {key} must be a single 2D field, got {array.shape}')

            return array

    raise KeyError(f'{path}: expected one of {keys}, found {list(arrays)}')


def load_example_fields(dates, generation_dir=None, *, baseline_dirs=None,
                        baseline_layout='legacy_physical', reference_rtol=0.001, reference_atol=0.001):
    """Load selected YYYYMMDD fields, in physical mm/day; never fall back to model space.

    Returns dates[date] with fields, ensemble [M,H,W], land and common finite valid
    mask. Saved PMM is retained; mean/median are computed from physical members.
    Pass DEFAULT_BASELINES explicitly to include original QM and bilinear fields.
    Baseline layout is 'legacy_physical' (pmm/lr_hr) or 'physical' (*_phys).
    Also works on an explicit revision sigma_star=<value> generation directory.
    """

    root = Path(generation_dir if generation_dir is not None else DEFAULT_GENERATION).expanduser().resolve()

    if baseline_layout not in ('legacy_physical', 'physical'):
        raise ValueError('Choose baseline_layout legacy_physical or physical')
    dates = [dates] if isinstance(dates, str) else list(dates)

    if not dates or len(set(dates)) != len(dates):
        raise ValueError('Supply one or more unique YYYYMMDD dates')

    result: dict[str, Any] = dict(directory=root, dates={}, sources={}, metadata={}, units='mm/day',
                                  units_basis='saved physical artifacts; no inference from magnitudes', origin='saved_generation')

    def read(path):
        result['sources'][str(path)] = str(path)
        return _npz(path)

    for date in dates:
        if len(date) != 8 or not date.isdigit():
            raise ValueError(f'Expected YYYYMMDD, got {date!r}')

        datetime.strptime(date, '%Y%m%d')

        path = root/'lr_hr_phys'/f'{date}.npz'

        refs = read(path)
        hr = _field(refs, ('hr',), path)
        lr = _field(refs, ('lr', 'lr_lrspace'), path)
        path = root/'ensembles_phys'/f'{date}.npz'
        ens = read(path)['ens']

        if ens.ndim == 4 and ens.shape[1] == 1:
            ens = ens[:, 0]
        if ens.ndim != 3 or ens.shape[0] == 0 or ens.shape[1:] != hr.shape:
            raise ValueError(f'{path}: expected [members, H, W], got {ens.shape}')

        path = root/'pmm_phys'/f'{date}.npz'
        fields = dict(danra=hr, era5_condition=lr, ceddar_mean=ens.mean(axis=0),
                      ceddar_median=np.median(ens, axis=0), ceddar_pmm=_field(read(path), ('pmm',), path))

        for method, directory in (baseline_dirs or {}).items():
            if method not in ('era5_bilinear', 'qm'):
                raise ValueError(f'Unsupported baseline method: {method}')

            directory = Path(directory).expanduser().resolve()
            suffix = '_phys' if baseline_layout == 'physical' else ''
            path = directory/f'lr_hr{suffix}'/f'{date}.npz'
            obs = _field(read(path), ('hr', 'HR', 'prcp_hr', 'PRCP_HR', 'obs', 'OBS'), path)

            if obs.shape != hr.shape or not np.allclose(obs, hr, rtol=reference_rtol,
                                                        atol=reference_atol, equal_nan=True):
                raise ValueError(f'{path}: baseline DANRA reference differs from CEDDAR')

            path = directory/f'pmm{suffix}'/f'{date}.npz'
            fields[method] = _field(read(path), ('pmm', 'PMM', 'gen', 'GEN', 'qm', 'QM', 'prcp', 'PRCP'), path)

        path = root/'lsm'/f'{date}.npz'
        if not path.is_file():
            path = root/'meta/land_mask.npz'

        land = _field(read(path), ('lsm_hr', 'lsm', 'mask'), path)
        if any(field.shape != hr.shape for field in (*fields.values(), land)):
            raise ValueError(f'{date}: field/mask grid shapes differ; no automatic regridding')

        valid = np.isfinite(land) & (land > 0.5) & np.isfinite(ens).all(axis=0)
        for field in fields.values():
            valid &= np.isfinite(field)

        result['dates'][date] = dict(fields=fields, ensemble=ens, land=land > 0.5, valid=valid)

    _metadata(result, root)
    return result
