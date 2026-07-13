#!/usr/bin/env python3
"""
Offline ML Training Script
Run this ONCE before starting Phase 3 real-time system.
Reads CSV files, trains models, saves to disk.

Usage:
    python3 train_models.py --data_dir ~/lunar_fdd_ws/data/raw-dynamic \
                            --model_dir ~/lunar_fdd_ws/src/hybrid_fdd/models

Architecture note:
- Trained models are loaded by MLClassifier at runtime
- Retrain when new data is collected or fault params change
- StandardScaler handles normalization across different severity ranges
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    mean_absolute_error, r2_score
)
from scipy import stats
from collections import deque
import sys

# Add hybrid_fdd to path for imports
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'hybrid_fdd'
))
from feature_extractor import (
    FeatureExtractor, FEATURE_NAMES, ANOMALY_FEATURE_INDICES
)
from kalman_filter import KalmanFilter


def load_and_merge_csvs(data_dir: str):
    """Load all sensor CSVs and merge with fault labels"""
    print(f'Loading data from {data_dir}...')

    all_data = []

    # Get all sensor files
    sensor_files = sorted(glob.glob(
        os.path.join(data_dir, '*sensors*.csv')
    ))

    for sensor_file in sensor_files:
        # Find matching fault file
        fault_file = sensor_file.replace('sensors', 'faults')

        if not os.path.exists(fault_file):
            print(f'  WARNING: No fault file for {sensor_file}, skipping')
            continue

        print(f'  Loading: {os.path.basename(sensor_file)}')

        # Load both files
        sensors_df = pd.read_csv(sensor_file)
        faults_df = pd.read_csv(fault_file)

        # Merge on nearest timestamp
        sensors_df = sensors_df.sort_values('timestamp')
        faults_df = faults_df.sort_values('timestamp')

        merged = pd.merge_asof(
            sensors_df,
            faults_df[['timestamp', 'fault_type',
                       'severity', 'is_active']],
            on='timestamp',
            direction='nearest',
            suffixes=('', '_fault')
        )

        # Clean fault labels
        merged['fault_type'] = merged['fault_type'].fillna('none')
        merged['severity'] = merged['severity'].fillna(0.0)
        # Tag every row with its source run so windows can be grouped by run
        merged['source_run'] = os.path.basename(sensor_file)

        all_data.append(merged)
        print(f'    Rows: {len(merged)} | '
              f'Faults: {merged["fault_type"].value_counts().to_dict()}')

    combined = pd.concat(all_data, ignore_index=True)
    print(f'\nTotal samples loaded: {len(combined)}')
    print(f'Fault distribution:\n{combined["fault_type"].value_counts()}')

    return combined


def extract_features_from_dataframe(df, window_size=100):
    """
    Extract windowed statistical features, one run at a time.

    Each run gets a fresh sliding window, so a window never spans two runs, and
    every window is tagged with its source run. That run tag lets the train/test
    split keep whole runs together (see train_and_save_models), which prevents
    the 50%-overlapping windows of a single run from leaking across the split.
    Joint velocities are clipped to physical limits to drop finite-difference
    spikes from the kinematically-controlled wrists.
    """
    print(f'\nExtracting features (window={window_size}, 50% overlap, per-run)...')

    X, y, severities, groups = [], [], [], []
    step = window_size // 2  # 50% overlap

    if 'source_run' not in df.columns:
        df = df.copy()
        df['source_run'] = 'run0'

    for run_name, run_df in df.groupby('source_run'):
        run_df = run_df.sort_values('timestamp').reset_index(drop=True)
        extractor = FeatureExtractor(window_size)
        # Fresh Kalman filter per run, exactly as the live node runs one filter
        # per experiment. Its position residuals populate the residual features;
        # this MUST match hybrid_fdd_node.py or the models see skewed inputs.
        kalman = KalmanFilter(n_joints=6)

        for i in range(len(run_df)):
            row = run_df.iloc[i]

            # Build mock snapshot (raw velocities; add_sample clips them)
            class MockSnap:
                pass
            snap = MockSnap()
            snap.joint_positions = [row[f'j{j}_pos'] for j in range(6)]
            snap.joint_velocities = [row[f'j{j}_vel'] for j in range(6)]
            snap.joint_efforts = [row[f'j{j}_eff'] for j in range(6)]
            snap.imu_linear_accel_x = row.get('imu_ax', 0.0)
            snap.imu_linear_accel_y = row.get('imu_ay', 0.0)
            snap.imu_linear_accel_z = row.get('imu_az', 0.0)
            snap.imu_angular_vel_x = row.get('imu_wx', 0.0)
            snap.imu_angular_vel_y = row.get('imu_wy', 0.0)
            snap.imu_angular_vel_z = row.get('imu_wz', 0.0)

            residuals, _ = kalman.process(
                snap.joint_positions, snap.joint_velocities
            )
            extractor.add_sample(snap, residuals)

            # Extract at step intervals once window is full
            if extractor.is_ready() and i % step == 0:
                features = extractor.extract_features()
                if features is not None:
                    X.append(features)
                    y.append(row['fault_type'])
                    severities.append(row.get('severity', 0.0))
                    groups.append(run_name)

    X = np.array(X)
    y = np.array(y)
    severities = np.array(severities)
    groups = np.array(groups)

    print(f'Feature matrix shape: {X.shape}')
    print(f'Label distribution: {pd.Series(y).value_counts().to_dict()}')
    print(f'Runs (groups): {len(set(groups))}')

    return X, y, severities, groups


def train_and_save_models(X, y, severities, groups, model_dir: str):
    """
    Train Isolation Forest + SVM + severity regressor and save to disk.
    StandardScaler handles normalization across different severity ranges.
    """
    os.makedirs(model_dir, exist_ok=True)

    print(f'\nTraining models...')

    # --- Label encoding ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'  Classes: {list(label_encoder.classes_)}')

    # --- Train/test split: hold out one whole run per experiment type ---
    # Whole runs go to train OR test, never both (no window leakage). We hold out
    # exactly one run of each experiment type so every class is represented in
    # the test set. (StratifiedGroupKFold does not work here: fault runs are
    # label-mixed -- each has a pre-fault 'none' segment -- so per-window
    # stratification can leave a whole fault class out of the fold.)
    EXPERIMENT_TYPES = ['baseline', 'bearing_wear',
                        'joint_stiffness', 'sensor_noise']

    def run_type(run_name):
        for t in EXPERIMENT_TYPES:
            if run_name.startswith(t):
                return t
        return 'unknown'

    unique_runs = sorted(set(groups))
    test_runs = set()
    for t in EXPERIMENT_TYPES:
        runs_t = [r for r in unique_runs if run_type(r) == t]
        if runs_t:
            test_runs.add(runs_t[-1])  # deterministic: last run of each type

    test_mask = np.array([g in test_runs for g in groups])
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # --- Normalization: fit the scaler on TRAIN ONLY ---
    # Fitting on all data would leak test statistics into the scaler. Fit on the
    # training windows only, then apply to everything. (This scaler is saved and
    # reused verbatim by the real-time node, so it must reflect training data.)
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_scaled = scaler.transform(X)
    print('  StandardScaler fitted on training runs only')

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    print(f'  Split: {len(train_idx)} train / {len(test_idx)} test windows')
    print(f'  Held-out test runs: {sorted(test_runs)}')

    # --- Isolation Forest (unsupervised anomaly detector) ---
    # Trained on HEALTHY TRAINING windows only, over the fault-bearing feature
    # subset (effort + IMU + residual). Position/velocity motion features are
    # excluded because healthy motion variance drowns the fault signal there.
    anom = ANOMALY_FEATURE_INDICES
    train_healthy_idx = train_idx[y[train_idx] == 'none']
    X_healthy = X_scaled[train_healthy_idx][:, anom]

    print(f'\n  Training Isolation Forest on {len(X_healthy)} healthy training '
          f'windows ({len(anom)} fault-bearing features)...')
    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Expect ~5% anomalies in healthy data
        random_state=42,
        n_jobs=-1
    )
    isolation_forest.fit(X_healthy)

    # Calibrate the binary anomaly threshold to a target false-positive rate on
    # healthy data. The default IF boundary (score < 0) and the old hardcoded
    # -0.1 almost never fired: fault windows are more anomalous than healthy but
    # still on the "normal" side in absolute terms. A percentile of the healthy
    # score distribution gives a usable operating point (tune the 10 as needed).
    healthy_scores = isolation_forest.decision_function(X_healthy)
    anomaly_threshold = float(np.percentile(healthy_scores, 10))  # ~10% FPR
    joblib.dump(anomaly_threshold,
                os.path.join(model_dir, 'anomaly_threshold.pkl'))
    print(f'  Anomaly threshold (~10% healthy FPR): {anomaly_threshold:.4f}')

    # Evaluate as a proper anomaly detector on the HELD-OUT test set:
    # ROC-AUC of fault(1)-vs-none(0) using the IF anomaly score. Unlike the old
    # contamination-pinned "accuracy", this actually moves with detector skill.
    test_scores = -isolation_forest.decision_function(
        X_scaled[test_idx][:, anom]
    )   # higher = more anomalous
    y_test_str = y[test_idx]
    test_binary = (y_test_str != 'none').astype(int)
    if test_binary.min() != test_binary.max():
        auc = roc_auc_score(test_binary, test_scores)
        print(f'  Isolation Forest fault-vs-none ROC-AUC (test): {auc:.3f}')
        for ft in ['bearing_wear', 'joint_stiffness', 'sensor_noise']:
            m = (y_test_str == ft) | (y_test_str == 'none')
            if (y_test_str[m] == ft).any():
                a = roc_auc_score(
                    (y_test_str[m] != 'none').astype(int), test_scores[m]
                )
                print(f'    {ft:16s} vs none AUC: {a:.3f}')
    else:
        print('  (test set has only one class; skipping IF ROC-AUC)')

    # Detection rate per fault at the calibrated threshold (fires if score < thr)
    test_decision = isolation_forest.decision_function(
        X_scaled[test_idx][:, anom]
    )
    print('  Anomaly-flag rates at calibrated threshold:')
    for ft in ['none', 'bearing_wear', 'joint_stiffness', 'sensor_noise']:
        m = y_test_str == ft
        if m.any():
            rate = float(np.mean(test_decision[m] < anomaly_threshold))
            tag = 'false-positive rate' if ft == 'none' else 'detect rate    '
            print(f'    {ft:16s} {tag}: {rate:.2f}')

    # --- SVM Classifier ---
    print(f'\n  Training SVM classifier on {len(X_train)} samples...')
    svm_classifier = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,     # Needed for confidence scores
        random_state=42,
        class_weight='balanced'   # Handle class imbalance
    )
    svm_classifier.fit(X_train, y_train)

    # Evaluate SVM
    y_pred = svm_classifier.predict(X_test)
    print('\n  SVM Classification Report:')
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))

    # --- Severity Regressor ---
    # Supervised regression of the true fault severity (0 for healthy, the
    # ramped value during a fault). Uses all 150 features and the already-
    # extracted `severities` labels. Unlike the unsupervised IF, this is
    # supervised so it can estimate bearing_wear severity from effort features.
    print(f'\n  Training severity regressor on {len(X_train)} samples...')
    severity_regressor = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    severity_regressor.fit(X_train, severities[train_idx])
    sev_pred = np.clip(severity_regressor.predict(X_test), 0.0, 1.0)
    sev_true = severities[test_idx]
    print(f'  Severity regressor MAE: {mean_absolute_error(sev_true, sev_pred):.3f}'
          f' | R2: {r2_score(sev_true, sev_pred):.3f}')
    for ft in ['bearing_wear', 'joint_stiffness', 'sensor_noise']:
        m = y_test_str == ft
        if m.any():
            print(f'    {ft:16s} MAE: '
                  f'{mean_absolute_error(sev_true[m], sev_pred[m]):.3f}')

    # --- Phase 4 Layer 2: supervised decision tree (cheap mid-tier) ---
    # Shallow tree on the fault-bearing subset. Fast (~0.1ms) and, being
    # supervised, it can screen bearing_wear (which the unsupervised IF cannot).
    # In the cascade it classifies when confident and escalates to Layer 3 (SVM)
    # when its class probability is below threshold.
    print(f'\n  Training Layer 2 decision tree on {len(X_train)} samples...')
    # Shallow tree with a leaf-size floor: deep/pure leaves give predict_proba
    # ~1.0 always, so confidence never falls below the escalation threshold and
    # Layer 3 stays unused. These give calibrated-ish confidence (leaves hold
    # class mixtures) -> uncertain windows escalate to the SVM, confident ones
    # resolve cheaply at Layer 2. Offline: kept-L2 acc ~0.93 vs escalated ~0.53.
    decision_tree = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=30,
        random_state=42, class_weight='balanced'
    )
    decision_tree.fit(X_train[:, anom], y_train)
    dt_pred = decision_tree.predict(X_test[:, anom])
    print('  Layer 2 tree report:')
    print(classification_report(
        y_test, dt_pred, target_names=label_encoder.classes_
    ))

    # --- Phase 4 Layer 1: statistical baselines (cheap health screen) ---
    # From healthy TRAINING windows, in RAW (unscaled) units, since the cascade's
    # Layer 1 works on raw rolling buffers. It compares a window's max |residual|
    # and per-joint effort RMS to these to decide "confidently healthy" vs escalate.
    resid_max_idx = [FEATURE_NAMES.index(f'j{j}_residual_max') for j in range(6)]
    eff_rms_idx = [FEATURE_NAMES.index(f'j{j}_eff_rms') for j in range(6)]
    Xh_raw = X[train_healthy_idx]
    per_window_maxresid = Xh_raw[:, resid_max_idx].max(axis=1)
    layer1_baseline = {
        # residual value treated as clearly anomalous (healthy mean + 4 sigma)
        'resid_scale': float(
            per_window_maxresid.mean() + 4.0 * per_window_maxresid.std()
        ),
        'eff_rms_mean': Xh_raw[:, eff_rms_idx].mean(axis=0).tolist(),
        'eff_rms_std': (Xh_raw[:, eff_rms_idx].std(axis=0) + 1e-6).tolist(),
    }
    print(f'  Layer 1 baseline: resid_scale='
          f'{layer1_baseline["resid_scale"]:.4f}')

    # --- Confusion Matrix Plot ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cmap='Blues'
    )
    plt.title('SVM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(model_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f'\n  Confusion matrix saved: {confusion_matrix_path}')

    # The node does single-sample inference, where n_jobs=-1 adds large
    # thread-spawn overhead per call (RF predict ~22ms vs ~5ms serial). Training
    # used all cores; force serial prediction before saving for the live node.
    isolation_forest.n_jobs = 1
    severity_regressor.n_jobs = 1

    # --- Save all models ---
    joblib.dump(isolation_forest,
                os.path.join(model_dir, 'isolation_forest.pkl'))
    joblib.dump(svm_classifier,
                os.path.join(model_dir, 'svm_classifier.pkl'))
    joblib.dump(scaler,
                os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoder,
                os.path.join(model_dir, 'label_encoder.pkl'))
    joblib.dump(severity_regressor,
                os.path.join(model_dir, 'severity_regressor.pkl'))
    joblib.dump(decision_tree,
                os.path.join(model_dir, 'decision_tree.pkl'))
    joblib.dump(layer1_baseline,
                os.path.join(model_dir, 'layer1_baseline.pkl'))

    print(f'\n  Models saved to: {model_dir}')
    print('  Files:')
    for f in os.listdir(model_dir):
        print(f'    {f}')

    return {
        'isolation_forest': isolation_forest,
        'svm': svm_classifier,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'test_accuracy': float(np.mean(y_pred == y_test))
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train hybrid FDD models'
    )
    parser.add_argument(
        '--data_dir',
        default=os.path.expanduser('~/lunar_fdd_ws/data/raw-dynamic'),
        help='Directory containing raw CSV files'
    )
    parser.add_argument(
        '--model_dir',
        default=os.path.expanduser(
            '~/lunar_fdd_ws/src/hybrid_fdd/models'
        ),
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=100,
        help='Sliding window size in samples (default: 100 = 1s at 100Hz)'
    )
    args = parser.parse_args()

    # Load data
    df = load_and_merge_csvs(args.data_dir)

    # Extract features
    X, y, severities, groups = extract_features_from_dataframe(
        df, args.window_size
    )

    # Train and save models
    results = train_and_save_models(
        X, y, severities, groups, args.model_dir
    )

    print(f'\n Training complete!')
    print(f'  Test accuracy: {results["test_accuracy"]:.3f}')
    print(f'  Models ready for real-time hybrid FDD node')


if __name__ == '__main__':
    main()