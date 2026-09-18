"""Strict adapter around legacy readers; inputs are read-only and must already be mm/day."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys

import numpy as np
import torch
import yaml

from baselines.evaluate_baselines.eval_dataresolver_baselines import BaselineDataResolver
from sbgm.evaluate.data_resolver import EvalDataResolver
from sbgm.provenance import git_info
from sbgm.runtime import external_output
from .common_masks import binary_mask, joint_land_mask

METHODS = ('era5_bilinear', 'qm', 'ceddar_mean', 'ceddar_median', 'ceddar_pmm')
BASELINES = {'era5_bilinear': 'bilinear', 'qm': 'qm'}
OBS_KEYS = ('hr', 'HR', 'prcp_hr', 'PRCP_HR', 'obs', 'OBS')
PRED_KEYS = ('pmm', 'PMM', 'gen', 'GEN', 'qm', 'QM', 'prcp', 'PRCP')


def date_string(value):
    date = str(value)
    if len(date) != 8 or not date.isdigit():
        raise ValueError(f'Expected YYYYMMDD date, got {value!r}')
    datetime.strptime(date, '%Y%m%d')
    return date


def calendar_dates():
    start = datetime(2019, 1, 1)
    return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(731)]


def resolve_path(value, parent):
    expanded = os.path.expandvars(str(value))
    if '$' in expanded:
        raise ValueError(f'Unresolved environment variable in path: {value}')
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def resolve_inputs(config_path, output_root=None, dates_file=None):
    """Resolve paths once; reject misspelled options instead of silently using defaults."""
    path = Path(config_path).resolve()
    cfg = yaml.safe_load(path.read_text())
    allowed = {'inputs', 'methods', 'output_root', 'expected_members', 'reference_rtol',
               'reference_atol', 'roi_mask', 'dates_file'}
    if not isinstance(cfg, dict) or set(cfg) - allowed:
        raise ValueError(f'Config must use only these keys: {sorted(allowed)}')
    cfg['methods'] = cfg.get('methods', list(METHODS))
    methods = cfg['methods']
    if not isinstance(methods, list) or not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f'Select unique methods from {METHODS}')
    members = cfg.get('expected_members', 32)
    if isinstance(members, bool) or not isinstance(members, int) or members < 2:
        raise ValueError('expected_members must be an integer >=2 (32 for manuscript evaluation)')
    cfg['expected_members'] = members
    for key in ('reference_rtol', 'reference_atol'):
        cfg[key] = float(cfg.get(key, 0.0))
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    inputs = cfg.get('inputs', {})
    required = {'ceddar'} | {BASELINES[m] for m in methods if m in BASELINES}
    if set(inputs) - {'ceddar', 'bilinear', 'qm'} or not required <= set(inputs):
        raise ValueError(f'Inputs must include {sorted(required)}; allowed: ceddar, bilinear, qm')
    for name, spec in inputs.items():
        if set(spec) - {'root', 'layout', 'units', 'units_evidence'}:
            raise ValueError(f'{name}: unknown input settings')
        if spec.get('units') != 'mm/day' or not str(spec.get('units_evidence', '')).strip():
            raise ValueError(f'{name}: declare units: mm/day and explain units_evidence')
        layout = spec.get('layout')
        if layout not in ('physical', 'legacy_physical') or (name == 'ceddar' and layout != 'physical'):
            raise ValueError(f'{name}: explicit physical layout required; CEDDAR needs *_phys folders')
        spec['root'] = str(resolve_path(spec['root'], path.parent))
    cfg['output_root'] = str(external_output(resolve_path(output_root or cfg['output_root'], path.parent)))
    for key, override in (('dates_file', dates_file), ('roi_mask', None)):
        value = override or cfg.get(key)
        cfg[key] = str(resolve_path(value, path.parent)) if value else None
    return cfg


def as_array(value, name):
    if value is None:
        raise ValueError(f'Missing required array: {name}')
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f'{name}: expected a numerical precipitation array')
    return array.astype(np.float64, copy=False)


def load_mask_array(path, shape):
    # Explicit key selection avoids the legacy NumPy "array or array" bug.
    with np.load(path, allow_pickle=False) as data:
        keys = [key for key in ('lsm_hr', 'lsm', 'mask', 'roi') if key in data.files]
        if not keys:
            raise ValueError(f'No land/ROI mask key in {path}')
        mask = data[keys[0]]
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    return binary_mask(mask, shape)


class RevisionInputs:
    """Load one date at a time, preserving field definitions and recording source files."""
    def __init__(self, config):
        self.config = config
        self.methods = config['methods']
        self.sources, self.files, self.shape = {}, {}, None
        required = ['ceddar'] + [BASELINES[m] for m in self.methods if m in BASELINES]
        for name in required:
            spec = config['inputs'][name]
            root = Path(spec['root'])
            if not root.is_dir():
                raise FileNotFoundError(f'{name}: missing input directory {root}')
            physical = spec['layout'] == 'physical'
            suffix = '_phys' if physical else ''
            artifacts = {'observation': (root / f'lr_hr{suffix}', ('hr',) if name == 'ceddar' else OBS_KEYS)}
            if name == 'ceddar':
                artifacts['ensemble'] = (root / 'ensembles_phys', ('ens',))
                if 'ceddar_pmm' in self.methods:
                    artifacts['prediction'] = (root / 'pmm_phys', ('pmm',))
                resolver = EvalDataResolver(root, eval_land_only=True, prefer_phys=True)
            else:
                artifacts['prediction'] = (root / f'pmm{suffix}', PRED_KEYS)
                resolver = BaselineDataResolver(root, prefer_phys=physical)
            for directory, _ in artifacts.values():
                if not directory.is_dir():
                    raise FileNotFoundError(f'{name}: required folder {directory}; model-space fallback is disabled')
            self.sources[name] = dict(root=root, artifacts=artifacts, resolver=resolver)
            for pattern in ('meta/*.json', 'meta/*.yaml', 'meta/*.yml', 'resolved_config.yaml'):
                for metadata in sorted(root.glob(pattern)):
                    self.record_file(metadata, name, 'source_metadata', '')
        roots = [source['root'] for source in self.sources.values()]
        if len(set(roots)) != len(roots):
            raise ValueError('Different input methods must have distinct roots')
        if config['dates_file']:
            self.record_file(Path(config['dates_file']), 'evaluation', 'frozen_dates', '')
        self.inventory = self.inventory_dates()
        self.dates = self.inventory['common_dates']

    def record_file(self, path, source, role, date):
        path = Path(path).resolve()
        before = path.stat()
        if str(path) in self.files:
            old = self.files[str(path)]
            if (before.st_size, before.st_mtime_ns) != (old['size_bytes'], old['mtime_ns']):
                raise ValueError(f'Input changed during evaluation: {path}')
            return
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Input changed while hashing: {path}')
        self.files[str(path)] = dict(source=source, role=role, date=date, path=str(path),
                                    size_bytes=after.st_size, mtime_ns=after.st_mtime_ns,
                                    sha256=digest.hexdigest())

    def inventory_dates(self):
        expected = set(calendar_dates())
        available, outside = {}, {}
        for name, source in self.sources.items():
            for role, (directory, _) in source['artifacts'].items():
                dates = {date_string(p.stem) for p in directory.glob('*.npz')}
                label = f'{name}/{role}'
                available[label] = dates & expected
                outside[label] = sorted(dates - expected)
        common = sorted(set.intersection(*available.values()))
        if not common:
            raise ValueError('No complete common 2019–2020 dates across required artifacts')
        if self.config['dates_file']:
            frozen = [date_string(line.strip()) for line in Path(self.config['dates_file']).read_text().splitlines() if line.strip()]
            if frozen != common:
                raise ValueError('Frozen date list differs from the common 2019–2020 artifact dates')
        return dict(period=['20190101', '20201231'], calendar_count=731, common_dates=common,
                    n_dates=len(common), missing_calendar_dates=sorted(expected - set(common)),
                    missing_by_artifact={k: sorted(expected-v) for k, v in available.items()},
                    outside_period_by_artifact=outside)

    def _check_artifacts(self, name, source, date):
        for role, (directory, keys) in source['artifacts'].items():
            path = directory / f'{date}.npz'
            self.record_file(path, name, role, date)
            with np.load(path, allow_pickle=False) as data:
                if not any(key in data.files for key in keys):
                    raise ValueError(f'{path}: missing required key from {keys}; fallback is disabled')

    def _land(self, name, source, date, shape):
        path = source['root'] / 'lsm' / f'{date}.npz'
        if not path.exists():
            path = source['root'] / 'meta/land_mask.npz'
        if not path.is_file():
            raise FileNotFoundError(f'{name}/{date}: a land mask is required')
        self.record_file(path, name, 'land_mask', date if path.parent.name == 'lsm' else '')
        return load_mask_array(path, shape)

    def load_date(self, date):
        if date not in self.dates:
            raise ValueError(f'Date is outside the frozen comparison: {date}')
        source = self.sources['ceddar']
        self._check_artifacts('ceddar', source, date)
        resolver = source['resolver']
        obs = as_array(resolver.load_obs(date), f'DANRA/{date}')
        if obs.ndim != 2 or not all(obs.shape):
            raise ValueError(f'{date}: expected nonempty 2-D DANRA field, got {obs.shape}')
        if self.shape is not None and self.shape != obs.shape:
            raise ValueError(f'{date}: spatial dimensions changed across dates')
        self.shape = obs.shape
        ens = as_array(resolver.load_ens(date), f'ensemble/{date}')
        if ens.shape != (self.config['expected_members'], *obs.shape):
            raise ValueError(f'{date}: expected {self.config["expected_members"]} members on {obs.shape}, got {ens.shape}')
        land = self._land('ceddar', source, date, obs.shape)
        fields = {}
        if 'ceddar_mean' in self.methods:
            fields['ceddar_mean'] = ens.mean(axis=0)
        if 'ceddar_median' in self.methods:
            fields['ceddar_median'] = np.quantile(ens, 0.5, axis=0, method='linear')
        if 'ceddar_pmm' in self.methods:
            fields['ceddar_pmm'] = as_array(resolver.load_pmm(date), f'PMM/{date}')
        for method, name in BASELINES.items():
            if method not in self.methods:
                continue
            source = self.sources[name]
            self._check_artifacts(name, source, date)
            other_obs = as_array(source['resolver'].load_obs(date), f'{name}/DANRA/{date}')
            if other_obs.shape != obs.shape or not np.array_equal(np.isfinite(obs), np.isfinite(other_obs)):
                raise ValueError(f'{name}/{date}: DANRA reference shape or finite support differs')
            finite = np.isfinite(obs)
            if not np.allclose(obs[finite], other_obs[finite],
                    rtol=self.config['reference_rtol'], atol=self.config['reference_atol']):
                raise ValueError(f'{name}/{date}: DANRA reference values differ; inspect grid/crop/units')
            if not np.array_equal(land, self._land(name, source, date, obs.shape)):
                raise ValueError(f'{name}/{date}: land mask differs from CEDDAR')
            fields[method] = as_array(source['resolver'].load_pmm(date), f'{method}/{date}')
        if self.config['roi_mask']:
            path = Path(self.config['roi_mask'])
            self.record_file(path, 'evaluation', 'roi_mask', '')
            land = land & load_mask_array(path, obs.shape)
        for method, field in fields.items():
            if field.shape != obs.shape:
                raise ValueError(f'{method}/{date}: prediction shape {field.shape} differs from DANRA {obs.shape}')
        valid = joint_land_mask(land, obs, ens, *fields.values())
        return dict(date=date, observation=obs, ensemble=ens, fields={m: fields[m] for m in self.methods},
                    land=land, valid=valid)

    def check_unchanged(self):
        for row in self.files.values():
            stat = Path(row['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (row['size_bytes'], row['mtime_ns']):
                raise ValueError(f'Input changed during evaluation: {row["path"]}')


def write_table(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if isinstance(v, (float, np.floating)) and np.isnan(v) else v
                             for k, v in row.items()})


def write_run_metadata(path, config, **details):
    manifest = dict(timestamp=datetime.now(timezone.utc).isoformat(), git=git_info(),
                    hostname=socket.gethostname(), command=sys.argv,
                    python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
                    device='cpu', torch_threads=torch.get_num_threads(), config=config,
                    units_status='declared in configuration; no magnitude-based unit inference',
                    mask_policy='common finite land across selected methods, DANRA and all ensemble members',
                    alignment_check='matching saved DANRA references and binary masks; no coordinate regridding',
                    median_definition='linear 0.5 quantile of physical ensemble members',
                    pmm_definition='saved physical PMM, unchanged; producer uses model-space PMM then inverse transform',
                    csv_nan='empty numeric cell; consult counts', **details)
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
