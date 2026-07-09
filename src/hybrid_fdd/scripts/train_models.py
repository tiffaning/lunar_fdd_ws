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
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from collections import deque
import sys

# Add hybrid_fdd to path for imports
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), '..', 'hybrid_fdd'
))
from feature_extractor import FeatureExtractor, FEATURE_NAMES


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

        all_data.append(merged)
        print(f'    Rows: {len(merged)} | '
              f'Faults: {merged["fault_type"].value_counts().to_dict()}')

    combined = pd.concat(all_data, ignore_index=True)
    print(f'\nTotal samples loaded: {len(combined)}')
    print(f'Fault distribution:\n{combined["fault_type"].value_counts()}')

    return combined


def extract_features_from_dataframe(df, window_size=100):
    """
    Extract windowed statistical features from full dataframe.
    Uses 50% overlap between windows.
    """
    print(f'\nExtracting features (window={window_size}, 50% overlap)...')

    X, y, severities = [], [], []
    step = window_size // 2  # 50% overlap

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    extractor = FeatureExtractor(window_size)

    for i in range(len(df)):
        row = df.iloc[i]

        # Build mock snapshot
        class MockSnap:
            pass
        snap = MockSnap()
        snap.joint_positions = [
            row['j0_pos'], row['j1_pos'], row['j2_pos'],
            row['j3_pos'], row['j4_pos'], row['j5_pos']
        ]
        snap.joint_velocities = [
            row['j0_vel'], row['j1_vel'], row['j2_vel'],
            row['j3_vel'], row['j4_vel'], row['j5_vel']
        ]
        snap.joint_efforts = [
            row['j0_eff'], row['j1_eff'], row['j2_eff'],
            row['j3_eff'], row['j4_eff'], row['j5_eff']
        ]
        snap.imu_linear_accel_x = row.get('imu_ax', 0.0)
        snap.imu_linear_accel_y = row.get('imu_ay', 0.0)
        snap.imu_linear_accel_z = row.get('imu_az', 0.0)
        snap.imu_angular_vel_x = row.get('imu_wx', 0.0)
        snap.imu_angular_vel_y = row.get('imu_wy', 0.0)
        snap.imu_angular_vel_z = row.get('imu_wz', 0.0)

        extractor.add_sample(snap)

        # Extract at step intervals once window is full
        if extractor.is_ready() and i % step == 0:
            features = extractor.extract_features()
            if features is not None:
                X.append(features)
                y.append(row['fault_type'])
                severities.append(row.get('severity', 0.0))

    X = np.array(X)
    y = np.array(y)
    severities = np.array(severities)

    print(f'Feature matrix shape: {X.shape}')
    print(f'Label distribution: {pd.Series(y).value_counts().to_dict()}')

    return X, y, severities


def train_and_save_models(X, y, model_dir: str):
    """
    Train Isolation Forest + SVM and save to disk.
    StandardScaler handles normalization across different severity ranges.
    """
    os.makedirs(model_dir, exist_ok=True)

    print(f'\nTraining models...')

    # --- Normalization ---
    # Critical: bearing_wear goes 0.1→0.8, sensor_noise 0.05→0.6
    # StandardScaler normalizes all features to zero mean, unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('  StandardScaler fitted (handles different severity ranges)')

    # --- Label encoding ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'  Classes: {list(label_encoder.classes_)}')

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # --- Isolation Forest ---
    # Train on healthy data ONLY to learn normal behavior
    healthy_mask = y == 'none'
    X_healthy = X_scaled[healthy_mask]

    print(f'\n  Training Isolation Forest on {len(X_healthy)} healthy samples...')
    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Expect ~5% anomalies in healthy data
        random_state=42,
        n_jobs=-1
    )
    isolation_forest.fit(X_healthy)

    # Evaluate IF on all data
    if_predictions = isolation_forest.predict(X_scaled)
    if_labels = np.where(y == 'none', 1, -1)  # 1=normal, -1=anomaly
    if_accuracy = np.mean(if_predictions == if_labels)
    print(f'  Isolation Forest accuracy: {if_accuracy:.3f}')

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

    # --- Save all models ---
    joblib.dump(isolation_forest,
                os.path.join(model_dir, 'isolation_forest.pkl'))
    joblib.dump(svm_classifier,
                os.path.join(model_dir, 'svm_classifier.pkl'))
    joblib.dump(scaler,
                os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoder,
                os.path.join(model_dir, 'label_encoder.pkl'))

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
    X, y, severities = extract_features_from_dataframe(
        df, args.window_size
    )

    # Train and save models
    results = train_and_save_models(X, y, args.model_dir)

    print(f'\n Training complete!')
    print(f'  Test accuracy: {results["test_accuracy"]:.3f}')
    print(f'  Models ready for real-time hybrid FDD node')


if __name__ == '__main__':
    main()