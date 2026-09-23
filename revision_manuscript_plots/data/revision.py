"""Read revision CSVs/manifests; preserve dates, member IDs, subsets and NaNs.

No metric recomputation, pooling of members, or changes to scientific definitions.
CSV rows remain strings, as in the existing revision plotting helpers.
"""
from pathlib import Path
from typing import cast

from ..paths import REVISION_ROOT
from .legacy import _bundle, _metadata, load_sigma_star as _sigma_tables


def _stage(stage, directory, required):
    """Load a specific stage of the revision evaluation, e.g. 'deterministic', 'dry_bias', 'probabilistic', or 'morphology'."""

    root = Path(directory if directory is not None else REVISION_ROOT/'evaluation'/stage).expanduser().resolve()
    manifest = root/'manifest.json'

    if not manifest.is_file():
        raise FileNotFoundError(f'Missing {manifest}; pass the raw {stage} evaluation folder')

    # Read the manifest before loading potentially large daily/member tables.
    import json
    info = json.loads(manifest.read_text())

    if info.get('stage') != stage or info.get('status') != 'complete':
        raise ValueError(f'{manifest}: expected complete {stage} evaluation')

    result = _bundle(root, required)
    result.update(origin='revision', manifest=info) # type: ignore
    dates = root/'dates.txt'

    if dates.is_file():
        result['dates'] = dates.read_text().splitlines() # type: ignore
        sources = cast(dict[str, str], result['sources'])
        sources['dates.txt'] = str(dates)

    return result


def load_deterministic(directory=None):
    """Daily paired errors, event contingency counts/scores and occurrence metrics."""
    return _stage('deterministic', directory,
                  ('daily_continuous_metrics.csv', 'event_detection_metrics.csv', 'occurrence_metrics.csv'))


def load_dry_bias(directory=None):
    """Own-wet-mask intensity and occurrence; retain per-member seasonal statistics.

    Optional member tables are returned when present, never replaced by PMM/mean.
    Missing member tables mean those panels cannot be drawn from that run.
    """
    return _stage('dry_bias', directory, ('conditional_intensity.csv', 'seasonal_decomposition.csv'))


def load_probabilistic(directory=None):
    """Revision CRPS, coverage, reliability, rank counts and binned spread-skill.

    Rank counts are NOT continuous PIT values. Binned spread-skill is NOT a daily
    scatterplot. Use legacy.load_probabilistic explicitly for those legacy panels.
    """
    return _stage('probabilistic', directory,
                  ('crps.csv', 'coverage_summary.csv', 'reliability.csv', 'spread_skill.csv', 'rank_histogram.csv'))


def load_morphology(directory=None):
    """Raw date/member absolute objects, equal-area controls and physical-threshold SAL.

    Preserve zero-object counts and undefined sizes/SAL separately. Plot aggregation
    belongs in the panel code (including abs-before-member-mean for count MAE).
    """
    return _stage('morphology', directory,
                  ('objects_absolute.csv', 'objects_equal_area.csv', 'sal_absolute.csv'))


def load_sigma_star(run_dir):
    """Load one explicitly selected revision run, e.g. .../legacy or .../ramp_0.30_0.55.

    Never pick the newest run or merge modes. Require a unique model evaluation;
    attach resolved_config.yaml and preserve recorded generation/sampler metadata.
    """

    run = Path(run_dir).expanduser().resolve()

    if not (run/'resolved_config.yaml').is_file():
        raise FileNotFoundError(f'Missing {run / "resolved_config.yaml"}')
    candidates = sorted(run.glob('evaluation/*/prcp/sigma_control'))

    if len(candidates) != 1:
        raise ValueError(f'{run}: expected one model sigma_control directory, found {len(candidates)}')

    result = _sigma_tables(candidates[0])
    _metadata(result, run)
    result = cast(dict, result)
    result.update(origin='revision', run_dir=run)

    return result
