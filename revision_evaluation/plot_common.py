"""Small shared helpers for table-only figures and external, exclusive output folders."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sbgm.provenance import git_info
from sbgm.runtime import external_output


def read_table(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key]) if row[key] else float('nan')


def prepare_output(input_dir, output_dir, manifest, stage):
    if manifest.get('status') != 'complete' or manifest.get('stage') != stage:
        raise ValueError(f'Plotting requires a complete {stage} manifest')
    output = external_output(output_dir)
    roots = [Path(input_dir).resolve()] + [Path(v['root']).resolve()
                                         for v in manifest['config']['inputs'].values()]
    if any(output == root or root in output.parents for root in roots):
        raise ValueError('Figure output cannot be inside metric or generation inputs')
    output.mkdir(parents=True, exist_ok=False)
    return output


def save_figure(fig, output, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def write_plot_metadata(output, files, source_manifest, script, **details):
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    record = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                  git=git_info(), evaluation_git=source_manifest['git'],
                  inputs={str(Path(p).resolve()): digest(p) for p in files},
                  script_sha256=digest(script), helper_sha256=digest(__file__),
                  matplotlib=matplotlib.__version__, numpy=np.__version__,
                  n_dates=source_manifest['n_dates'], n_pixel_days=source_manifest['n_pixel_days'],
                  **details)
    (output / 'figure_provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def style():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.axisbelow': True})
