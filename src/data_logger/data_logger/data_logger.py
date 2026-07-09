#!/usr/bin/env python3
import csv
import os
import time
from datetime import datetime


class DataLogger:
    def __init__(self, experiment_name: str, log_dir: str = '/home/tiffa/lunar_fdd_ws/data/raw-dynamic'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(log_dir, exist_ok=True)

        self.sensor_file = os.path.join(
            log_dir, f'{experiment_name}_sensors_{self.session_id}.csv'
        )
        self.fault_file = os.path.join(
            log_dir, f'{experiment_name}_faults_{self.session_id}.csv'
        )
        self.energy_file = os.path.join(
            log_dir, f'{experiment_name}_energy_{self.session_id}.csv'
        )
        self._init_csv_files()
        print(f'[DataLogger] Logging to: {log_dir}')

    def _init_csv_files(self):
        with open(self.sensor_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'j0_pos', 'j1_pos', 'j2_pos', 'j3_pos', 'j4_pos', 'j5_pos',
                'j0_vel', 'j1_vel', 'j2_vel', 'j3_vel', 'j4_vel', 'j5_vel',
                'j0_eff', 'j1_eff', 'j2_eff', 'j3_eff', 'j4_eff', 'j5_eff',
                'imu_ax', 'imu_ay', 'imu_az',
                'imu_wx', 'imu_wy', 'imu_wz',
                'fault_active', 'fault_type'
            ])

        with open(self.fault_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'fault_type', 'affected_joint',
                'severity', 'progression_rate', 'is_active'
            ])

        with open(self.energy_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cpu_percent', 'memory_mb',
                'estimated_energy_joules', 'processing_component',
                'detection_layer'
            ])

    def log_sensor_data(self, joint_positions, joint_velocities,
                        joint_efforts, imu_data, fault_active, fault_type):
        with open(self.sensor_file, 'a', newline='') as f:
            writer = csv.writer(f)
            pos = list(joint_positions) + [0.0] * (6 - len(joint_positions))
            vel = list(joint_velocities) + [0.0] * (6 - len(joint_velocities))
            eff = list(joint_efforts) + [0.0] * (6 - len(joint_efforts))
            writer.writerow([
                time.time(),
                *pos[:6], *vel[:6], *eff[:6],
                imu_data.get('ax', 0.0), imu_data.get('ay', 0.0),
                imu_data.get('az', 0.0), imu_data.get('wx', 0.0),
                imu_data.get('wy', 0.0), imu_data.get('wz', 0.0),
                fault_active, fault_type
            ])

    def log_fault_event(self, fault_type, affected_joint,
                        severity, progression_rate, is_active):
        with open(self.fault_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), fault_type, affected_joint,
                severity, progression_rate, is_active
            ])

    def log_energy_metrics(self, cpu_percent, memory_mb,
                           energy_joules, component,
                           detection_layer='none'):
        with open(self.energy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), cpu_percent, memory_mb,
                energy_joules, component, detection_layer
            ])
