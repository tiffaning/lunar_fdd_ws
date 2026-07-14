#!/usr/bin/env python3
"""
Phase 5 evaluation: compare the continuous hybrid FDD vs the model cascade.

Reads the per-run CSVs written by fdd_evaluator_node (eval_<strategy>_<exp>_*.csv)
and computes, per run, the full metric set, then aggregates across replications,
runs significance tests, and (optionally) an offline threshold grid search.

Usage:
    python3 evaluate_phase5.py --log_dir ~/lunar_fdd_ws/data/phase5
    python3 evaluate_phase5.py --log_dir ~/lunar_fdd_ws/data/phase5 \
            --grid_dir ~/lunar_fdd_ws/data/phase5_grid

Outputs (to --out_dir, default <log_dir>/results):
    summary.csv           per (strategy, experiment) mean +/- std of every metric
    significance.csv      Welch t-test (continuous vs cascade) + one-way ANOVA
    *.png                 bar charts + cascade layer usage + grid Pareto (if matplotlib)
Everything is also printed to stdout for the results section.
"""
import argparse
import csv
import glob
import os
import numpy as np
from scipy import stats

EXPERIMENTS = ['baseline', 'bearing_wear', 'joint_stiffness', 'sensor_noise']
STRATEGIES = ['continuous', 'cascade']
WARMUP_S = 20.0   # skip startup transients but keep a clean pre-fault window
                  # (fault onset is ~t=28s; tune this post-hoc, analysis-only)

METRIC_ORDER = [
    'accuracy', 'false_alarm_rate', 'detection_rate', 'time_to_detection_s',
    'severity_mae', 'mean_proc_ms', 'mean_energy_j', 'mean_cpu', 'mean_mem_mb',
    'pct_layer1', 'pct_layer2', 'pct_layer3'
]


def load_run(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['t_rel'] = float(r['t_rel'])
        r['severity_est'] = float(r['severity_est'])
        r['proc_ms'] = float(r['proc_ms'])
        r['layer'] = int(float(r['layer']))
        r['gt_severity'] = float(r['gt_severity'])
        r['gt_is_active'] = str(r['gt_is_active']).lower() == 'true'
        r['cpu_percent'] = float(r['cpu_percent'])
        r['mem_mb'] = float(r['mem_mb'])
        r['energy_j'] = float(r['energy_j'])
    return [r for r in rows if r['t_rel'] >= WARMUP_S]


def run_metrics(rows):
    if not rows:
        return None
    det = np.array([r['det_fault_type'] for r in rows])
    gt = np.array([r['gt_fault_type'] for r in rows])
    active = np.array([r['gt_is_active'] for r in rows])
    m = {}
    m['accuracy'] = float(np.mean(det == gt))
    none_mask = gt == 'none'
    m['false_alarm_rate'] = (float(np.mean(det[none_mask] != 'none'))
                             if none_mask.any() else np.nan)
    # detection rate = recall while the fault is actually active
    fa = active & (gt != 'none')
    m['detection_rate'] = (float(np.mean(det[fa] == gt[fa]))
                           if fa.any() else np.nan)
    # time-to-detection: fault onset -> first correct call
    if fa.any():
        t = np.array([r['t_rel'] for r in rows])
        onset = t[fa].min()
        correct = fa & (det == gt)
        m['time_to_detection_s'] = (float(t[correct].min() - onset)
                                    if correct.any() else np.nan)
    else:
        m['time_to_detection_s'] = np.nan
    # severity error on correctly-detected fault windows
    sev_mask = (det == gt) & (gt != 'none')
    if sev_mask.any():
        se = np.array([r['severity_est'] for r in rows])[sev_mask]
        gs = np.array([r['gt_severity'] for r in rows])[sev_mask]
        m['severity_mae'] = float(np.mean(np.abs(se - gs)))
    else:
        m['severity_mae'] = np.nan
    m['mean_proc_ms'] = float(np.mean([r['proc_ms'] for r in rows]))
    m['mean_energy_j'] = float(np.mean([r['energy_j'] for r in rows]))
    m['mean_cpu'] = float(np.mean([r['cpu_percent'] for r in rows]))
    m['mean_mem_mb'] = float(np.mean([r['mem_mb'] for r in rows]))
    layers = np.array([r['layer'] for r in rows])
    for k in (1, 2, 3):
        m[f'pct_layer{k}'] = float(np.mean(layers == k)) * 100.0
    return m


def collect(log_dir):
    """-> {(strategy, experiment): [run_metrics, ...]}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(log_dir, 'eval_*.csv'))):
        base = os.path.basename(path)
        parts = base[len('eval_'):].rsplit('_', 2)  # strategy_exp_date_time
        # strategy is first token; experiment is the middle (may contain '_')
        toks = base[len('eval_'):].split('_')
        strat = toks[0]
        # experiment = everything between strategy and the trailing date_time
        exp = '_'.join(toks[1:-2])
        if strat not in STRATEGIES:
            continue
        mm = run_metrics(load_run(path))
        if mm is None:
            continue
        out.setdefault((strat, exp), []).append(mm)
    return out


def agg(vals):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)])
    if v.size == 0:
        return (np.nan, np.nan, 0)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', default=os.path.expanduser(
        '~/lunar_fdd_ws/data/phase5'))
    ap.add_argument('--grid_dir', default='')
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.log_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)

    data = collect(args.log_dir)
    if not data:
        print(f'No eval_*.csv found in {args.log_dir}')
        if args.grid_dir:
            grid_search(args.grid_dir, out_dir)   # grid can run standalone
        return

    # --- Summary table ---
    print('\n===== PER-CONDITION SUMMARY (mean +/- std over replications) =====')
    summary_rows = []
    for exp in EXPERIMENTS:
        for strat in STRATEGIES:
            runs = data.get((strat, exp))
            if not runs:
                continue
            n = len(runs)
            print(f'\n{strat:10s} | {exp:15s} | {n} runs')
            row = {'strategy': strat, 'experiment': exp, 'n_runs': n}
            for met in METRIC_ORDER:
                mean, std, k = agg([r[met] for r in runs])
                row[f'{met}_mean'] = mean
                row[f'{met}_std'] = std
                if not np.isnan(mean):
                    print(f'    {met:22s} {mean:8.3f} +/- {std:.3f}')
            summary_rows.append(row)

    # write summary.csv
    if summary_rows:
        keys = ['strategy', 'experiment', 'n_runs'] + \
            [f'{m}_{s}' for m in METRIC_ORDER for s in ('mean', 'std')]
        with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: r.get(k, '') for k in keys})

    # --- Significance: continuous vs cascade per experiment/metric (Welch) ---
    print('\n===== SIGNIFICANCE: continuous vs cascade (Welch t-test) =====')
    sig_rows = []
    n_tests = 0
    for exp in EXPERIMENTS:
        c = data.get(('continuous', exp))
        k = data.get(('cascade', exp))
        if not c or not k:
            continue
        for met in METRIC_ORDER:
            a = np.array([r[met] for r in c if not np.isnan(r[met])])
            b = np.array([r[met] for r in k if not np.isnan(r[met])])
            if a.size < 2 or b.size < 2:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)
            sig_rows.append({'experiment': exp, 'metric': met,
                             'continuous_mean': a.mean(), 'cascade_mean': b.mean(),
                             't': t, 'p': p})
            n_tests += 1
    # Bonferroni
    alpha = 0.05
    bonf = alpha / max(n_tests, 1)
    print(f'  {n_tests} tests | Bonferroni-corrected alpha = {bonf:.5f}')
    for r in sig_rows:
        sigflag = '*' if r['p'] < bonf else ' '
        print(f"  {sigflag} {r['experiment']:15s} {r['metric']:20s} "
              f"cont={r['continuous_mean']:.3f} casc={r['cascade_mean']:.3f} "
              f"p={r['p']:.4g}")
    if sig_rows:
        with open(os.path.join(out_dir, 'significance.csv'), 'w',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()) +
                               ['bonferroni_alpha'])
            w.writeheader()
            for r in sig_rows:
                r['bonferroni_alpha'] = bonf
                w.writerow(r)

    # --- One-way ANOVA across the 4 scenarios (per strategy, per metric) ---
    print('\n===== ANOVA across scenarios (per strategy) =====')
    for strat in STRATEGIES:
        for met in ['mean_energy_j', 'accuracy', 'mean_proc_ms']:
            groups = []
            for exp in EXPERIMENTS:
                runs = data.get((strat, exp))
                if runs:
                    g = [r[met] for r in runs if not np.isnan(r[met])]
                    if len(g) >= 2:
                        groups.append(g)
            if len(groups) >= 2:
                F, p = stats.f_oneway(*groups)
                print(f'  {strat:10s} {met:18s} F={F:.3f} p={p:.4g}')

    _plots(data, out_dir)
    if args.grid_dir:
        grid_search(args.grid_dir, out_dir)
    print(f'\nResults written to {out_dir}')


def _plots(data, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('  (matplotlib unavailable; skipping plots)')
        return
    for met, title in [('accuracy', 'Detection Accuracy'),
                       ('mean_proc_ms', 'Compute Time (ms/detection)'),
                       ('mean_energy_j', 'Energy (J/detection)'),
                       ('false_alarm_rate', 'False Alarm Rate')]:
        x = np.arange(len(EXPERIMENTS))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, strat in enumerate(STRATEGIES):
            means = []
            errs = []
            for exp in EXPERIMENTS:
                runs = data.get((strat, exp))
                if runs:
                    m, s, _ = agg([r[met] for r in runs])
                else:
                    m, s = np.nan, 0
                means.append(m)
                errs.append(s)
            ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3,
                   label=strat)
        ax.set_xticks(x)
        ax.set_xticklabels(EXPERIMENTS, rotation=15)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{met}.png'))
        plt.close(fig)

    # cascade layer usage (stacked)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(EXPERIMENTS))
    for k in (1, 2, 3):
        vals = []
        for exp in EXPERIMENTS:
            runs = data.get(('cascade', exp))
            vals.append(agg([r[f'pct_layer{k}'] for r in runs])[0]
                        if runs else 0)
        vals = np.nan_to_num(vals)
        ax.bar(EXPERIMENTS, vals, bottom=bottoms, label=f'Layer {k}')
        bottoms += vals
    ax.set_title('Cascade layer usage (%)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'cascade_layers.png'))
    plt.close(fig)


def grid_search(grid_dir, out_dir):
    """Offline replay of the cascade gating over an l1/l2/l2_none grid, using
    record_*.csv (cascade record_all_layers mode). Produces an accuracy-vs-energy
    Pareto frontier so the operating point can be justified."""
    files = sorted(glob.glob(os.path.join(grid_dir, 'record_*.csv')))
    if not files:
        print(f'\n  (no record_*.csv in {grid_dir}; skipping grid search)')
        return
    rows = []
    for path in files:
        with open(path) as f:
            for r in csv.DictReader(f):
                if float(r['t_rel']) < WARMUP_S:
                    continue
                rows.append(r)
    if not rows:
        return
    l1c = np.array([float(r['l1_conf']) for r in rows])
    l2p = np.array([r['l2_pred'] for r in rows])
    l2c = np.array([float(r['l2_conf']) for r in rows])
    l3p = np.array([r['l3_pred'] for r in rows])
    gt = np.array([r['gt_type'] for r in rows])
    t1 = np.array([float(r['t1_ms']) for r in rows])
    t2 = np.array([float(r['t2_ms']) for r in rows])
    t3 = np.array([float(r['t3_ms']) for r in rows])

    print('\n===== THRESHOLD GRID SEARCH (offline replay) =====')
    grid = np.arange(0.1, 1.0, 0.1)
    results = []
    for l1 in grid:
        for l2 in grid:
            for l2n in grid:
                pred = np.empty(len(rows), dtype=object)
                cost = np.empty(len(rows))
                stop1 = l1c >= l1
                pred[stop1] = 'none'
                cost[stop1] = t1[stop1]
                rest = ~stop1
                is_fault = l2p != 'none'
                keep2 = rest & (((is_fault) & (l2c >= l2)) |
                                ((~is_fault) & (l2c >= l2n)))
                pred[keep2] = l2p[keep2]
                cost[keep2] = t1[keep2] + t2[keep2]
                go3 = rest & ~keep2
                pred[go3] = l3p[go3]
                cost[go3] = t1[go3] + t2[go3] + t3[go3]
                acc = float(np.mean(pred == gt))
                results.append((l1, l2, l2n, acc, float(cost.mean())))
    # Pareto frontier (maximize accuracy, minimize cost)
    results.sort(key=lambda r: (-r[3], r[4]))
    pareto = []
    best_cost = np.inf
    for r in sorted(results, key=lambda r: r[4]):   # by cost asc
        if r[3] > (pareto[-1][3] if pareto else -1):
            pareto.append(r)
    with open(os.path.join(out_dir, 'grid_search.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['l1', 'l2', 'l2_none', 'accuracy', 'mean_cost_ms'])
        for r in results:
            w.writerow([f'{r[0]:.1f}', f'{r[1]:.1f}', f'{r[2]:.1f}',
                        f'{r[3]:.4f}', f'{r[4]:.3f}'])
    print('  Pareto frontier (accuracy vs mean compute cost):')
    for r in pareto:
        print(f'    l1={r[0]:.1f} l2={r[1]:.1f} l2n={r[2]:.1f} -> '
              f'acc={r[3]:.3f}  cost={r[4]:.2f}ms')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([r[4] for r in results], [r[3] for r in results],
                   s=8, alpha=0.3, label='all combos')
        ax.plot([r[4] for r in pareto], [r[3] for r in pareto],
                'r-o', label='Pareto frontier')
        ax.set_xlabel('mean compute cost (ms/detection)')
        ax.set_ylabel('accuracy')
        ax.set_title('Cascade threshold trade-off')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'grid_pareto.png'))
        plt.close(fig)
    except Exception:
        pass


if __name__ == '__main__':
    main()
