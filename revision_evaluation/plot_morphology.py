"""Compact Group 4 diagnostics from saved tables, with ensemble members nested in dates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from .common_io import write_table
from .plot_common import (plt, np, read_table, number, prepare_output,
                          save_figure, write_plot_metadata, style)

METHODS = {'era5_bilinear': ('Bilinear ERA5', '#555555', 'o'),
           'qm': ('QM', '#C48100', 's'),
           'ceddar_members': ('CEDDAR members', '#0072B2', 'D')}


def date_summaries(rows, metrics, analysis, paired=False):
    """One value per date/method; count MAE takes abs before averaging members."""
    groups = defaultdict(list)
    reference = {}
    for row in rows:
        date, q = row['date'], number(row, 'reference_threshold')
        if row['method'] == 'danra':
            reference[date, q] = row
        else:
            groups[date, row['method'], q].append(row)
    result = []
    for (date, method, q), group in groups.items():
        for metric in metrics:
            count_error = metric == 'object_count_absolute_error'
            key = 'n_objects' if count_error else metric
            values = np.array([number(row, key) for row in group])
            if paired:
                values = values-number(reference[date, q], key)
            if count_error:
                values = np.abs(values)
            values = values[np.isfinite(values)]
            quantiles = np.quantile(values, [.25, .5, .75]) if values.size else [np.nan]*3
            result.append(dict(date=date, method=method, reference_threshold=q, analysis=analysis,
                               metric=metric, member_q25=float(quantiles[0]),
                               value=float(values.mean()) if count_error and values.size else float(quantiles[1]),
                               member_q75=float(quantiles[2]), n_values=len(group), n_valid_values=len(values),
                               reference_n_objects=number(reference[date, q], 'n_objects') if paired else np.nan,
                               fraction_with_count_error=float((values > 0).mean()) if count_error and values.size else np.nan))
    return result


def common_dates(rows):
    """Compare methods on the same defined dates for each plotted metric/threshold."""
    dates = [{r['date'] for r in rows if r['method'] == method and np.isfinite(r['value'])}
             for method in METHODS]
    return set.intersection(*dates)


def count_event_summary(summaries):
    """Separate event-day errors from false alarms; each date has equal weight."""
    counts = [r for r in summaries if r['metric'] == 'object_count_absolute_error']
    result = []
    for analysis, q in sorted({(r['analysis'], r['reference_threshold']) for r in counts}):
        selected = [r for r in counts if r['analysis'] == analysis and r['reference_threshold'] == q]
        dates = common_dates(selected)
        for cohort in ('all', 'reference_event', 'reference_event_free'):
            for method in METHODS:
                group = [r for r in selected if r['method'] == method and r['date'] in dates
                         and (cohort == 'all' or
                              (cohort == 'reference_event' and r['reference_n_objects'] > 0) or
                              (cohort == 'reference_event_free' and r['reference_n_objects'] == 0))]
                result.append(dict(analysis=analysis, reference_threshold=q, cohort=cohort, method=method,
                    n_dates=len(group), min_valid_members=min((r['n_valid_values'] for r in group), default=0),
                    mean_absolute_count_error=float(np.mean([r['value'] for r in group])) if group else np.nan,
                    # On reference-free dates, any predicted object is a false alarm.
                    false_alarm_frequency=float(np.mean([r['fraction_with_count_error'] for r in group]))
                        if group and cohort == 'reference_event_free' else np.nan))
    return result


def run(input_dir, output_dir=None, sal_threshold=1.):
    input_dir = Path(input_dir).resolve()
    files = [input_dir/name for name in ('objects_absolute.csv', 'objects_equal_area.csv', 'sal_absolute.csv')]
    absolute, relative, sal = [read_table(path) for path in files]
    manifest_file = input_dir/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    thresholds = manifest['thresholds']
    if sal_threshold not in thresholds:
        raise ValueError(f'SAL threshold must be one of {thresholds}')
    output = prepare_output(input_dir, output_dir or input_dir.parent/'manuscript_ready/morphology', manifest, 'morphology')
    summaries = []
    for rows, analysis in ((absolute, 'absolute'), (relative, 'equal_area')):
        summaries.extend(date_summaries(rows, ['n_objects', 'object_count_absolute_error',
                                              'largest_object_fraction'], analysis, paired=True))
    summaries.extend(date_summaries(relative, ['wet_fraction_difference'], 'equal_area_matching'))
    summaries.extend(date_summaries(sal, ['S', 'A', 'L'], 'sal'))
    write_table(output/'date_level_summary.csv', summaries)
    write_table(output/'object_count_summary.csv', count_event_summary(summaries))
    style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.86, wspace=.30, hspace=.65)
    plot_counts = []
    for ax, analysis, metric, letter in zip(axes.ravel(),
            ('absolute', 'absolute', 'equal_area', 'equal_area'),
            ('object_count_absolute_error', 'largest_object_fraction',
             'object_count_absolute_error', 'largest_object_fraction'), 'abcd'):
        count_error = metric == 'object_count_absolute_error'
        selected = [r for r in summaries if r['analysis'] == analysis and r['metric'] == metric]
        cohorts = {q: common_dates([r for r in selected if r['reference_threshold'] == q]) for q in thresholds}
        for method, (label, colour, marker) in METHODS.items():
            centres, lower, upper = [], [], []
            for q in thresholds:
                group = [r for r in selected if r['method'] == method and r['reference_threshold'] == q
                         and r['date'] in cohorts[q]]
                reduce_dates = np.mean if count_error else np.median
                centres.append(float(reduce_dates([r['value'] for r in group])) if group else np.nan)
                # CEDDAR band: median of the per-date member quartiles, not pooled members.
                lower.append(float(np.median([r['member_q25'] for r in group])) if group else np.nan)
                upper.append(float(np.median([r['member_q75'] for r in group])) if group else np.nan)
                plot_counts.append(dict(analysis=analysis, metric=metric, method=method,
                                        reference_threshold=q, n_dates=len(group),
                                        min_valid_members=min((r['n_valid_values'] for r in group), default=0)))
            ax.plot(thresholds, centres, color=colour, marker=marker, label=label)
            if method == 'ceddar_members' and not count_error:
                ax.fill_between(thresholds, lower, upper, color=colour, alpha=.18)
        ax.axhline(0, color='.4', linestyle='--', linewidth=1)
        name = 'Absolute thresholds' if analysis == 'absolute' else 'Equal-area control'
        ax.set_title(f'({letter}) {name}', loc='left')
        ax.set_ylabel('Mean absolute object-count error' if count_error else 'Largest-object fraction − DANRA')
        if count_error:
            ax.set_ylim(bottom=0)
        ax.set_xticks(thresholds, [f'{q:g}\nn={len(cohorts[q])}' for q in thresholds])
        ax.set_xlabel('DANRA reference threshold (mm/day)')
        ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .15), ncol=3, frameon=False)
    fig.suptitle('Precipitation morphology: comparison with DANRA', y=.98, fontsize=13)
    fig.text(.5, .925, 'Counts: mean absolute errors. Largest-object fractions: median signed differences.', ha='center', fontsize=10)
    fig.text(.5, .025, 'CEDDAR: members summarized within each date, then dates weighted equally.\n'
             'Blue bands in (b, d): median member quartiles across dates; not confidence intervals.\n'
             'n: common defined dates. Count panels include event-free dates; see object_count_summary.csv.\n'
             'Equal-area ties are retained; inspect achieved fractions in the tables.',
             ha='center', fontsize=9, linespacing=1.5)
    save_figure(fig, output, 'group4_objects')

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.77, wspace=.30)
    for ax, metric, title in zip(axes, ('S', 'A', 'L'), ('Structure', 'Amplitude', 'Location')):
        selected = [r for r in summaries if r['analysis'] == 'sal' and r['metric'] == metric
                    and r['reference_threshold'] == sal_threshold]
        dates = common_dates(selected)
        groups = [[r['value'] for r in selected if r['method'] == method and r['date'] in dates] for method in METHODS]
        boxes = ax.boxplot(groups, patch_artist=True, showfliers=True,
                            labels=['Bilinear\nERA5', 'QM', 'CEDDAR\nmembers'])
        for box, (_, colour, _) in zip(boxes['boxes'], METHODS.values()):
            box.set(facecolor=colour, alpha=.5)
        ax.axhline(0, color='.4', linestyle='--', linewidth=1)
        ax.set_title(f'{title} ({metric})\nn={len(dates)} common dates')
        ax.set_ylim((-2.05, 2.05) if metric != 'L' else (-.03, 2.05))
        for method in METHODS:
            group = [r for r in selected if r['method'] == method and r['date'] in dates]
            plot_counts.append(dict(analysis='sal', metric=metric, method=method,
                                    reference_threshold=sal_threshold, n_dates=len(group),
                                    min_valid_members=min((r['n_valid_values'] for r in group), default=0)))
    fig.suptitle(f'SAL components with physical object threshold ≥{sal_threshold:g} mm/day', fontsize=13)
    fig.text(.5, .10, 'CEDDAR: one member median per date. Boxes: IQR across dates; all outliers shown.\n'
             'Undefined components excluded on common dates; valid member counts saved in the summary tables.',
             ha='center', fontsize=9, linespacing=1.5)
    save_figure(fig, output, f'group4_sal_q{sal_threshold:g}')
    write_table(output/'plot_sample_counts.csv', plot_counts)
    write_plot_metadata(output, files+[manifest_file], manifest, __file__, sal_threshold=sal_threshold,
                        aggregation='members within dates, then dates; never pool members as independent dates',
                        object_count='mean across dates of within-date member mean absolute count errors',
                        object_count_cohorts='all, reference_event, reference_event_free; common defined dates',
                        object_fraction='median across dates of within-date member median signed differences',
                        object_band='largest-object fraction only: median within-date member quartiles across common finite dates',
                        sal_boxes='IQR across dates of within-date member medians')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sal-threshold', type=float, default=1.)
    args = parser.parse_args()
    print(run(args.input_dir, args.output_dir, args.sal_threshold))


if __name__ == '__main__':
    main()
