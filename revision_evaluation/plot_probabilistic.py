"""Plot saved Group 2 tables only; no arrays, model loading or new metric calculations."""
import argparse
import json
from pathlib import Path

from .plot_common import (plt, np, read_table, number, prepare_output,
                          save_figure, write_plot_metadata, style)

SUBSETS = {'all_land': ('All land', '#0072B2'),
           'observed_wet': ('Observed wet (DANRA ≥1 mm/day)', '#D55E00')}


def run(input_dir, output_dir=None):
    input_dir = Path(input_dir).resolve()
    names = ['coverage_summary.csv', 'reliability.csv', 'spread_skill.csv',
             'rank_histogram.csv', 'crps.csv']
    files = [input_dir / name for name in names]
    manifest = json.loads((input_dir / 'manifest.json').read_text())
    tables = {name: read_table(path) for name, path in zip(names, files)}
    output = prepare_output(input_dir, output_dir or input_dir.parent / 'manuscript_ready/probabilistic',
                            manifest, 'probabilistic')
    style()
    subtitle = f"{manifest['n_dates']} dates; {manifest['config']['expected_members']} ensemble members"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout='constrained')
    for ax, metric, title, ylabel in zip(axes,
            ('empirical_coverage', 'mean_interval_width'),
            ('(a) Central interval coverage', '(b) Central interval width'),
            ('Empirical coverage', 'Mean interval width (mm/day)')):
        for subset, (label, colour) in SUBSETS.items():
            rows = sorted([r for r in tables['coverage_summary.csv']
                           if r['season'] == 'ALL' and r['subset'] == subset and r['metric'] == metric],
                          key=lambda r: number(r, 'nominal_coverage'))
            x = [number(r, 'nominal_coverage') for r in rows]
            ax.plot(x, [number(r, 'median') for r in rows], 'o-', color=colour, label=label)
            ax.fill_between(x, [number(r, 'p25') for r in rows], [number(r, 'p75') for r in rows],
                            color=colour, alpha=.15)
        ax.set(title=title, xlabel='Nominal central coverage', ylabel=ylabel,
               xticks=manifest['coverages'], xlim=(.45, .95))
        ax.grid(alpha=.2)
    axes[0].plot([0, 1], [0, 1], '--', color='.5', linewidth=1)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc='best')
    fig.suptitle('Ensemble intervals: daily medians and interquartile ranges\n'+subtitle)
    save_figure(fig, output, 'coverage')

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), height_ratios=(3, 1.25), layout='constrained')
    for column, threshold in enumerate(manifest['thresholds']):
        rows = sorted([r for r in tables['reliability.csv'] if number(r, 'threshold') == threshold],
                      key=lambda r: number(r, 'bin'))
        ax, counts = axes[:, column]
        ax.plot([0, 1], [0, 1], '--', color='.5', linewidth=1)
        ax.plot([number(r, 'mean_forecast_probability') for r in rows],
                [number(r, 'observed_frequency') for r in rows], 'o-', color='#0072B2')
        ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025), title=f'Precipitation ≥{threshold:g} mm/day',
               xlabel='Mean forecast probability', ylabel='Observed frequency' if column == 0 else '')
        left = np.array([number(r, 'bin_left') for r in rows])
        widths = np.array([number(r, 'bin_right') for r in rows])-left
        counts.bar(left, [number(r, 'n_cases') for r in rows], width=widths*.9,
                   align='edge', color='#0072B2', alpha=.65)
        counts.set(xlim=(-.025, 1.025), xlabel='Forecast probability bin',
                   ylabel='Cases (pixel-days)' if column == 0 else '')
        counts.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        total = sum(int(r['n_cases']) for r in rows)
        events = sum(int(r['n_events']) for r in rows)
        ax.text(.04, .96, f'{events:,} observed events\n{total:,} cases', transform=ax.transAxes,
                va='top', fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Reliability and bin populations — all land\n'+subtitle)
    save_figure(fig, output, 'reliability')

    rows = tables['rank_histogram.csv']
    fig, ax = plt.subplots(figsize=(8, 3.6), layout='constrained')
    ax.bar([number(r, 'rank') for r in rows], [number(r, 'frequency') for r in rows],
           color='#0072B2', width=.85)
    ax.axhline(number(rows[0], 'expected_frequency'), color='.4', linestyle='--', label='Uniform reference')
    ax.set(xlabel='Observation rank (0 = below all members; M = above all members)',
           ylabel='Relative frequency', ylim=(0, None), title='Randomized rank histogram — all land\n'+subtitle)
    ax.legend(fontsize=9)
    save_figure(fig, output, 'rank_histogram')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    maximum = 0.0
    for subset, (label, colour) in SUBSETS.items():
        rows = sorted([r for r in tables['spread_skill.csv'] if r['subset'] == subset],
                      key=lambda r: number(r, 'bin'))
        x = np.array([number(r, 'mean_spread') for r in rows])
        y = np.array([number(r, 'rmse') for r in rows])
        finite = np.concatenate((x[np.isfinite(x)], y[np.isfinite(y)]))
        if finite.size:
            maximum = max(maximum, float(finite.max()))
        axes[0].plot(x, y, 'o-', color=colour, label=label)
        axes[1].plot([number(r, 'bin')+1 for r in rows], [number(r, 'n_cases') for r in rows],
                     'o-', color=colour, label=label)
    limit = maximum*1.05 if maximum else 1
    axes[0].plot([0, limit], [0, limit], '--', color='.5', linewidth=1)
    axes[0].set(xlim=(0, limit), ylim=(0, limit), xlabel='Mean ensemble spread (mm/day)',
                ylabel='Ensemble-mean RMSE (mm/day)', title='(a) Spread–skill')
    axes[1].set(xlabel='Pooled spread bin', ylabel='Cases (pixel-days)', title='(b) Bin populations', ylim=(0, None))
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].legend(fontsize=8)
    fig.suptitle('Common pooled spread bins; sample standard deviation (ddof=1)\n'+subtitle)
    save_figure(fig, output, 'spread_skill')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout='constrained')
    for ax, (subset, (label, colour)) in zip(axes, SUBSETS.items()):
        groups, labels = [], []
        for season in ('ALL', 'DJF', 'MAM', 'JJA', 'SON'):
            values = [number(r, 'crps') for r in tables['crps.csv'] if r['subset'] == subset
                      and (season == 'ALL' or r['season'] == season) and np.isfinite(number(r, 'crps'))]
            groups.append(values)
            labels.append(f'{season}\nn={len(values)}')
        boxes = ax.boxplot(groups, labels=labels, patch_artist=True, showfliers=True)
        for box in boxes['boxes']:
            box.set(facecolor=colour, alpha=.45)
        ax.set(title=label, ylabel='Daily empirical ensemble CRPS (mm/day)', ylim=(0, None))
    fig.suptitle('Daily CRPS distributions; boxes: IQR, line: median, all outliers shown\n'+subtitle)
    save_figure(fig, output, 'crps_daily')
    write_plot_metadata(output, files+[input_dir / 'manifest.json'], manifest, __file__,
                        variability='Coverage bands and CRPS boxes: across dates, not confidence intervals',
                        observed_wet='Conditional diagnostic; nominal/uniform calibration targets do not apply automatically')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    print(run(args.input_dir, args.output_dir))


if __name__ == '__main__':
    main()
