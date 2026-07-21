#!/usr/bin/env python3
"""
Regenerate the accuracy / energy / compute-time bar charts with significance
brackets and asterisks (continuous vs. cascade, per scenario). Writes NEW files
(<metric>_sig.png) so the originals are untouched.

Significance is recomputed from the Phase 5 run data with Welch's t-tests:
    *** p < 0.001,  ** p < 0.01,  * p < 0.05,  ns otherwise.

Usage:
    python3 plot_significance.py --log_dir ~/lunar_fdd_ws/data/phase5
"""
import argparse
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_phase5 import collect, EXPERIMENTS, STRATEGIES


# Bonferroni-corrected family-wise threshold (matches the paper: 0.05 / 45 tests).
# A comparison must clear this to be called significant; the number of asterisks
# then reflects magnitude. This keeps the figure consistent with Table 3, where
# e.g. baseline accuracy (p = 1.75e-3) is NOT significant.
BONF_ALPHA = 0.05 / 45  # = 0.00111

def stars(p):
    if p >= BONF_ALPHA:
        return 'ns'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    return '*'


def series(data, strat, metric):
    means, stds = [], []
    for exp in EXPERIMENTS:
        runs = data.get((strat, exp))
        v = np.array([r[metric] for r in runs
                      if not np.isnan(r[metric])]) if runs else np.array([])
        means.append(v.mean() if v.size else np.nan)
        stds.append(v.std(ddof=1) if v.size > 1 else 0.0)
    return np.array(means), np.array(stds)


def plot_metric(data, metric, title, ylabel, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = np.arange(len(EXPERIMENTS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    m = {s: series(data, s, metric) for s in STRATEGIES}
    for i, s in enumerate(STRATEGIES):
        means, stds = m[s]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=3, label=s)

    # headroom for the brackets
    tops = np.array([max(m['continuous'][0][j] + m['continuous'][1][j],
                         m['cascade'][0][j] + m['cascade'][1][j])
                     for j in range(len(EXPERIMENTS))])
    ymax = np.nanmax(tops)
    ax.set_ylim(0, ymax * 1.22)

    for j, exp in enumerate(EXPERIMENTS):
        c = data.get(('continuous', exp))
        k = data.get(('cascade', exp))
        a = np.array([r[metric] for r in c if not np.isnan(r[metric])]) if c else np.array([])
        b = np.array([r[metric] for r in k if not np.isnan(r[metric])]) if k else np.array([])
        if a.size < 2 or b.size < 2:
            continue
        _, p = stats.ttest_ind(a, b, equal_var=False)
        lab = stars(p)
        y = tops[j] + 0.05 * ymax
        h = 0.02 * ymax
        x1, x2 = j - 0.5 * w, j + 0.5 * w
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c='k')
        ax.text(j, y + h, lab, ha='center', va='bottom',
                fontsize=(11 if lab != 'ns' else 9),
                fontweight=('bold' if lab != 'ns' else 'normal'))

    ax.set_xticks(x)
    ax.set_xticklabels(EXPERIMENTS, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', default=os.path.expanduser(
        '~/lunar_fdd_ws/data/phase5'))
    args = ap.parse_args()
    out_dir = os.path.join(args.log_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    data = collect(args.log_dir)
    if not data:
        print(f'No eval_*.csv found in {args.log_dir}')
        return

    plot_metric(data, 'accuracy', 'Detection Accuracy', 'accuracy',
                os.path.join(out_dir, 'accuracy_sig.png'))
    plot_metric(data, 'mean_energy_j', 'Energy (J/detection)', 'energy (J)',
                os.path.join(out_dir, 'mean_energy_j_sig.png'))
    plot_metric(data, 'mean_proc_ms', 'Compute Time (ms/detection)',
                'compute time (ms)',
                os.path.join(out_dir, 'mean_proc_ms_sig.png'))
    plot_metric(data, 'false_alarm_rate', 'False Alarm Rate',
                'false alarm rate',
                os.path.join(out_dir, 'false_alarm_rate_sig.png'))


if __name__ == '__main__':
    main()
