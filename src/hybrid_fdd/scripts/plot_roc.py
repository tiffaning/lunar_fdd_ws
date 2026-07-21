#!/usr/bin/env python3
"""
Plot the Isolation Forest ROC curves (fault vs. healthy) from the trained model
and held-out test runs. Styled to match the other Phase 5 figures.

Usage:
    python3 plot_roc.py \
        --data_dir  ~/lunar_fdd_ws/data/raw-dynamic \
        --model_dir ~/lunar_fdd_ws/src/hybrid_fdd/models \
        --out       ~/lunar_fdd_ws/data/phase5/results/roc_curve.png
"""
import argparse
import glob
import os
import sys
import csv
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
from sklearn.metrics import roc_curve, roc_auc_score

# make the feature extractor / kalman importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hybrid_fdd'))
from feature_extractor import FeatureExtractor, ANOMALY_FEATURE_INDICES
from kalman_filter import KalmanFilter

FAULTS = ['bearing_wear', 'joint_stiffness', 'sensor_noise']


def extract_run(path):
    """Windowed features + per-window fault_type label for one run (matches
    the training pipeline: Kalman residuals, 100-sample windows, 50% overlap)."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    ex = FeatureExtractor(100)
    kf = KalmanFilter(6)
    X, y = [], []

    class Snap:
        pass

    for i, r in enumerate(rows):
        s = Snap()
        s.joint_positions = [float(r[f'j{j}_pos']) for j in range(6)]
        s.joint_velocities = [float(r[f'j{j}_vel']) for j in range(6)]
        s.joint_efforts = [float(r[f'j{j}_eff']) for j in range(6)]
        s.imu_linear_accel_x = float(r['imu_ax'])
        s.imu_linear_accel_y = float(r['imu_ay'])
        s.imu_linear_accel_z = float(r['imu_az'])
        s.imu_angular_vel_x = float(r['imu_wx'])
        s.imu_angular_vel_y = float(r['imu_wy'])
        s.imu_angular_vel_z = float(r['imu_wz'])
        res, _ = kf.process(s.joint_positions, s.joint_velocities)
        ex.add_sample(s, res)
        if ex.is_ready() and i % 50 == 0:
            X.append(ex.extract_features())
            y.append(r['fault_type'])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=os.path.expanduser(
        '~/lunar_fdd_ws/data/raw-dynamic'))
    ap.add_argument('--model_dir', default=os.path.expanduser(
        '~/lunar_fdd_ws/src/hybrid_fdd/models'))
    ap.add_argument('--out', default=os.path.expanduser(
        '~/lunar_fdd_ws/data/phase5/results/roc_curve.png'))
    args = ap.parse_args()

    scaler = joblib.load(os.path.join(args.model_dir, 'scaler.pkl'))
    iso = joblib.load(os.path.join(args.model_dir, 'isolation_forest.pkl'))
    anom = ANOMALY_FEATURE_INDICES

    # Held-out test runs = last (sorted) run of each condition, matching the
    # run-level split used in training.
    X, y = [], []
    for cond in ['baseline'] + FAULTS:
        f = sorted(glob.glob(
            os.path.join(args.data_dir, f'{cond}_sensors_*.csv')))[-1]
        Xr, yr = extract_run(f)
        X += Xr
        y += yr
    X = np.array(X)
    y = np.array(y)

    # Isolation Forest anomaly score (higher = more anomalous)
    scores = -iso.decision_function(scaler.transform(X)[:, anom])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    # overall fault-vs-none
    binary = (y != 'none').astype(int)
    fpr, tpr, _ = roc_curve(binary, scores)
    ax.plot(fpr, tpr, lw=2.2, color='k',
            label=f'overall (AUC = {roc_auc_score(binary, scores):.3f})')
    # per-fault: that fault vs. none
    for ft in FAULTS:
        m = (y == ft) | (y == 'none')
        yb = (y[m] == ft).astype(int)
        fpr, tpr, _ = roc_curve(yb, scores[m])
        ax.plot(fpr, tpr, lw=1.8,
                label=f'{ft} (AUC = {roc_auc_score(yb, scores[m]):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='chance')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Isolation Forest ROC (fault vs. healthy)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
