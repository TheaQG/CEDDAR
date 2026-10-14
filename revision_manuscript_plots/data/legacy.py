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

DEFAULT_BASELINES = (DEFAULT_EVALUATION.parent.parent / "baselines")
# DEFAULT_BASELINES = {name: GENERATION_ROOT / 'baselines' / directory / 'test'
                    #  for name, directory in [('era5_bilinear', 'bilinear'), ('qm', 'qm')]}


def baseline_evaluation(name: str) -> Path:
    """Return legacy precipitation-evaluation root for a baseline"""
    return(DEFAULT_BASELINES / name / "test" / "prcp")


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


def _normalise_date(date):
    """Return YYYYMMDD for supported saved date representations"""
    date = str(date)

    return datetime.strptime(date.replace("-", ""), "%Y%m%d").strftime("%Y%m%d")


def _season_indices(dates):
    """Return seasonal row indices for an array of saved dates"""
    dates = np.asarray([_normalise_date(date) for date in dates])
    months = np.array([int(date[4:6]) for date in dates])
    seasons = {'DJF': (12, 1, 2), 'MAM': (3, 4, 5), 'JJA': (6, 7, 8), 'SON': (9, 10, 11)}

    return {season: np.flatnonzero(np.isin(months, members,)) for season, members in seasons.items()}


def load_seasonal_distributions(directory=None):
    """Load daily histograms and return season row indices (no PDF renormalization)."""
    result = _evaluation('distributional', directory, ('dist_daily.npz',), 'dist_')

    dates = result['arrays']['dist_daily']['dates'].astype(str)

    result['season_indices'] = (_season_indices(dates))
    return result


def build_ensemble_histograms(dates, *, generation_dir=None, bins, land_only=True):
    """Build daily precipitation histograms for every saved CEDDAR member."""

    root = Path(generation_dir if generation_dir is not None else DEFAULT_GENERATION).expanduser().resolve()

    dates = np.asarray([_normalise_date(date) for date in dates])
    bins = np.asarray(bins, dtype=float,)

    if bins.ndim != 1 or len(bins) < 2:
        raise ValueError("bins must be a one-dimensional array of bin edges")

    if not np.all(np.diff(bins) > 0):
        raise ValueError("Histogram bin edges must be strictly increasing")

    # --------------------------------------------------------------
    # Determine ensemble/member shape from first date
    # --------------------------------------------------------------

    first_path = (root / "ensembles_phys" / f"{dates[0]}.npz")
    first = _npz(first_path)

    if "ens" not in first:
        raise KeyError(f"{first_path}: expected key 'ens', found {list(first)}")

    first_ens = np.asarray(first["ens"])

    if (first_ens.ndim == 4 and first_ens.shape[1] == 1):
        first_ens = first_ens[:, 0]

    if first_ens.ndim != 3:
        raise ValueError(f"{first_path}: expected [members, H, W] or [members, 1, H, W], got {first_ens.shape}")

    n_members, height, width = (first_ens.shape)
    n_bins = len(bins) - 1

    # uint32 is ample: an individual member contributes at most
    # H*W pixels to any daily histogram.
    counts_by_member = np.zeros((len(dates), n_members, n_bins,), dtype=np.uint32,)

    # --------------------------------------------------------------
    # Land mask
    # --------------------------------------------------------------

    common_land = None

    meta_mask = (root / "meta" / "land_mask.npz")

    if land_only and meta_mask.is_file():
        mask_arrays = _npz(meta_mask)

        common_land = _field(mask_arrays, ("lsm_hr", "lsm", "mask"), meta_mask,)

        if common_land.shape != (height, width,):
            raise ValueError(f"{meta_mask}: mask shape {common_land.shape} differs from ensemble grid {(height, width)}")

        common_land = (np.isfinite(common_land) & (common_land > 0.5))

    # --------------------------------------------------------------
    # Stream one date at a time
    # --------------------------------------------------------------

    for day_index, date in enumerate(dates):
        path = (root / "ensembles_phys" / f"{date}.npz")
        arrays = _npz(path)

        if "ens" not in arrays:
            raise KeyError(f"{path}: expected key 'ens', found {list(arrays)}")

        ens = np.asarray(arrays["ens"])

        if (ens.ndim == 4 and ens.shape[1] == 1):
            ens = ens[:, 0]

        if ens.shape != (n_members, height, width,):
            raise ValueError(f"{path}: expected  {(n_members, height, width)}, got {ens.shape}")

        # Some old generation layouts store a mask per date.
        land = common_land

        if land_only and land is None:
            mask_path = (root / "lsm" / f"{date}.npz")

            if not mask_path.is_file():
                raise FileNotFoundError(f"No common mask at {meta_mask} and no date mask at {mask_path}")

            mask_arrays = _npz(mask_path)
            land = _field(mask_arrays, ("lsm_hr", "lsm", "mask"), mask_path)

            if land.shape != (height, width,):
                raise ValueError(f"{mask_path}: mask shape {land.shape} differs from ensemble grid {(height, width)}")

            land = (np.isfinite(land) & (land > 0.5))

        for member in range(n_members):
            values = ens[member]

            if land_only:
                values = values[land]
            else:
                values = values.ravel()

            values = values[np.isfinite(values)]

            counts_by_member[day_index,member, ] = np.histogram(values, bins=bins,)[0]

    counts_pooled = (counts_by_member.sum(axis=1, dtype=np.uint64,))

    return {
        "dates": dates,
        "bins": bins,
        "counts_by_member": counts_by_member,
        "counts_pooled": counts_pooled,
        "season_indices": _season_indices(dates),
        "metadata": {
            "source": "saved_generation",
            "generation_directory": str(root),
            "ensemble_members": int(n_members),
            "land_only": bool(land_only),
            "histogram_basis": ("member x land-pixel values, retaining daily and member dimensions"),
        },
    }

def build_qm_histograms(dates, qm_generation_dir, bins, land_mask,):
    """Build land-only daily QM histograms from raw generated fields."""

    dates = [_normalise_date(date) for date in dates]

    bins = np.asarray(bins, dtype=float,)

    land = np.asarray(land_mask, dtype=bool,)

    counts = np.zeros((len(dates), len(bins) - 1,), dtype=np.int64,)

    for i, date in enumerate(dates):
        path = (Path(qm_generation_dir) / "pmm" / f"{date}.npz")

        arrays = _npz(path)

        field = _field(arrays,("pmm", "qm", "gen", "generated",), path,)
        field = np.asarray(field, dtype=float,).squeeze()

        if field.shape != land.shape:
            raise ValueError(f"{date}: QM field shape {field.shape} != land mask {land.shape}")

        values = field[land]
        values = values[np.isfinite(values)]

        counts[i], _ = np.histogram(values, bins=bins,)

    return {
        "dates": np.asarray(dates),
        "bins": bins,
        "counts": counts,
    }


def save_qm_histograms(product, path,):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True,)

    np.savez_compressed(path, **product,)


def load_qm_histograms(path):
    """Load corrected land-only QM daily histograms."""

    path = Path(path).expanduser().resolve()
    arrays = _npz(path)

    required = {"dates", "bins", "counts",}

    missing = required - set(arrays)

    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")

    dates = arrays["dates"].astype(str)
    bins = np.asarray(arrays["bins"], dtype=float,)
    counts = np.asarray(arrays["counts"])

    if counts.ndim != 2:
        raise ValueError(f"{path}: expected counts [date, bin], got {counts.shape}")

    if counts.shape[0] != len(dates):
        raise ValueError(f"{path}: {counts.shape[0]} histogram rows but {len(dates)} dates")

    if counts.shape[1] != len(bins) - 1:
        raise ValueError(f"{path}: histogram has {counts.shape[1]} bins, but bin edges imply {len(bins) - 1}")

    return {
        **arrays,
        "dates": dates,
        "bins": bins,
        "counts": counts,
        "season_indices": _season_indices(dates),
        "directory": path.parent,
        "source": str(path),
        "origin": "derived_land_only_qm_histograms",
    }
    

def save_ensemble_histograms(product, path,):
    """Save derived ensemble histogram product for manuscript plotting."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True,)

    np.savez_compressed(
        path,
        dates=product["dates"],
        bins=product["bins"],
        counts_by_member=product["counts_by_member"],
        counts_pooled=product["counts_pooled"],
    )


def load_ensemble_histograms(path,):
    """Load previously derived ensemble histogram product."""

    path = Path(path).expanduser().resolve()
    arrays = _npz(path)

    required = {
        "dates",
        "bins",
        "counts_by_member",
        "counts_pooled",
    }

    missing = (required - set(arrays))

    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")

    dates = (arrays["dates"].astype(str))

    return {
        **arrays,
        "dates": dates,
        "season_indices": (_season_indices(dates)),
        "directory": path.parent,
        "source": str(path),
        "origin": ("derived_from_saved_generation"),
    }


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


def load_land_mask(generation_dir=None):
    """Load saved HR land-sea mask."""

    root = Path(generation_dir if generation_dir is not None else DEFAULT_GENERATION).expanduser().resolve()

    candidates = (root / "meta" / "land_mask.npz", root / "lsm" / "land_mask.npz",)

    # First try explicitly named static products.
    for path in candidates:
        if not path.is_file():
            continue

        arrays = _npz(path)
        land = _field(arrays, ("lsm_hr", "lsm", "mask", "land",), path,)

        return (np.asarray(land, dtype=float,).squeeze() >= 0.5)

    # Otherwise use the first dated mask.
    lsm_dir = root / "lsm"

    if lsm_dir.is_dir():
        dated = sorted(lsm_dir.glob("*.npz"))

        if dated:
            path = dated[0]

            arrays = _npz(path)
            land = _field(arrays, ("lsm_hr", "lsm", "mask", "land",), path,)

            return (np.asarray(land, dtype=float,).squeeze() >= 0.5)

    raise FileNotFoundError(f"No land mask found below {root}")


def load_spatial_summary(path, *, field="sum"):
    """Load one saved legacy spatial precipitation summary."""

    path = Path(path).expanduser().resolve()
    arrays = _npz(path)

    if field not in arrays:
        raise ValueError(f'{path}: field "{field}" not found in saved arrays. Available fields: {sorted(arrays)}')

    values = np.asarray(arrays[field], dtype=float)

    if values.ndim != 2:
        raise ValueError(f'{path}: expected 2D array for field "{field}", got {values.ndim}D array')

    return {"field": field, "values": values, "path": path}


def load_dry_bias_spatial_maps(directory, *, qm_directory=None, derived_directory=None, field="sum", year="2019"):
    """Load saved legacy spatial maps used in the dry-bias figure"""

    directory = Path(directory).expanduser().resolve()

    if year not in ("2019", "2020"):
        raise ValueError(f'Unsupported year: {year}. Supported years are 2019 and 2020.')

    files = {
        "danra": directory / f"spatial_hr_{year}.npz",
        "era5_bilinear": directory / f"spatial_lr_{year}.npz",
        "ceddar_mean": directory / f"spatial_ensmean_{year}.npz",
    }

    # QM lives under own baseline eval
    if qm_directory is not None:
        qm_directory = Path(qm_directory).expanduser().resolve()

        qm_candidates = (
            qm_directory / f"spatial_pmm_{year}.npz",
            qm_directory / f"spatial_gen_{year}.npz",
            qm_directory / f"spatial_qm_{year}.npz",
        )

        qm_matches = [path for path in qm_candidates if path.is_file()]

        if len(qm_matches) != 1:
            raise FileNotFoundError(f'Expected exactly one QM spatial map in {qm_directory} for year {year}, found {len(qm_matches)}')
        files["qm"] = qm_matches[0]

    # Newly derived CEDDAR products
    if derived_directory is not None:
        derived_directory = Path(derived_directory).expanduser().resolve()
        files["ceddar_median"] = derived_directory / f"spatial_ens_median_{year}.npz"
        files["ceddar_pmm"] = derived_directory / f"spatial_pmm_{year}.npz"

    result = {}

    for method, path in files.items():
        product = load_spatial_summary(path, field=field,)
        result[method] = product["values"]

    return result


def build_ceddar_spatial_summaries(*, generation_dir, year="2019", dates=None,):
    """Derive spatial mean/sum maps for CEDDAR median and PMM."""
    root = Path(generation_dir).expanduser().resolve()

    if dates is None:
        dates = sorted(path.stem for path in (root / "ensembles_phys").glob(f"{year}*.npz"))
    else:
        dates = [_normalise_date(date) for date in dates if str(date).startswith(year)]

    if not dates:
        raise ValueError(f'No valid dates found for year {year} in {generation_dir}')

    median_fields = []
    pmm_fields = []

    for date in dates:
        ens_path = (root / "ensembles_phys" / f"{date}.npz")
        pmm_path = (root / "pmm_phys" / f"{date}.npz")

        ens_arrays = _npz(ens_path)

        if "ens" not in ens_arrays:
            raise KeyError(f"{ens_path}: expected key 'ens' not found")

        ens = np.asarray(ens_arrays["ens"], dtype=float)

        if (ens.ndim == 4 and ens.shape[1] == 1):
            ens = ens[:, 0]

        if ens.ndim != 3:
            raise ValueError(f"{ens_path}: expected [M, H, W], got {ens.shape}")

        daily_median = np.nanmedian(ens, axis=0)

        pmm_arrays = _npz(pmm_path)
        daily_pmm = _field(pmm_arrays, ("pmm",), pmm_path)

        if daily_median.shape != daily_pmm.shape:
            raise ValueError(f"Shape mismatch between daily median and PMM for date {date}: {daily_median.shape} vs {daily_pmm.shape}")

        median_fields.append(daily_median)
        pmm_fields.append(np.asarray(daily_pmm, dtype=float))

    median_stack = np.stack(median_fields, axis=0)
    pmm_stack = np.stack(pmm_fields, axis=0)

    return {
        "dates": np.asarray(dates),
        "ceddar_median": {
            "mean": np.nanmean(median_stack, axis=0),
            "sum": np.nansum(median_stack, axis=0)
        },
        "ceddar_pmm": {
            "mean": np.nanmean(pmm_stack, axis=0),
            "sum": np.nansum(pmm_stack, axis=0)
        }
    }


def save_ceddar_spatial_summaries(product, output_dir, *, year="2019",):
    """Save derived median and PMM spatial products"""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, stem in (("ceddar_median", "ens_median"), ("ceddar_pmm", "pmm")):
        maps = product[method]
        np.savez_compressed(output_dir / f"spatial_{stem}_{year}.npz", mean=maps["mean"], sum=maps["sum"])
        (output_dir / f"spatial_{stem}_{year}.meta.txt").write_text(f"source={stem}\ngroup={year}\nn_days={len(product['dates'])}\nderived_from=saved_physical_generation\n")
        