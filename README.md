# Lunar Construction Robot Fault Detection System

A hybrid (physics + machine learning) fault detection and diagnosis (FDD) system
for energy-efficient lunar construction robots, with a three-layer **model
cascade** that escalates from lightweight screening to the full hybrid pipeline
only when needed.

## Overview

The system addresses the energy–computation trade-off in FDD for lunar
manipulators. A full hybrid detector (Kalman filter + support vector machine +
random-forest severity estimation) is accurate but expensive to run
continuously. A staged cascade resolves most detections with cheap models and
invokes the full pipeline only for uncertain cases, cutting per-detection energy
while preserving most of the accuracy. The environment simulates lunar gravity
(1.62 m/s²) and a high-friction regolith surface (μ = 1.5) with a 6-DOF
Universal Robots UR10 arm.

## Project Status — complete

| Phase | Description | Status |
|---|---|---|
| 1 | Robot simulation environment (UR10, lunar world, hybrid controller) | ✅ |
| 2 | Fault injection + performance monitoring + data logging | ✅ |
| 3 | Hybrid FDD (Kalman residuals + IF + SVM + severity regressor) | ✅ |
| 4 | Three-layer model cascade | ✅ |
| 5 | Evaluation pipeline (continuous vs. cascade) + threshold grid search | ✅ |

## Key Results

- The cascade reduced per-detection compute time and estimated energy by
  **36–50%** across scenarios (all significant, p ≤ 1×10⁻³).
- Detection accuracy stayed within **0.6–7.2 percentage points** of the full
  hybrid system; the reduction was significant only for the mechanical faults
  (bearing wear, joint stiffness) and negligible for sensor noise.
- SVM fault classification: **0.98** test accuracy. Severity regressor: **MAE
  0.025**. Severity error was unchanged between continuous and cascade.
- The accuracy/energy trade-off is tunable via the escalation thresholds
  (selected point: l1 = 0.9, l2 = 0.9, l2_none = 0.7).

## System Architecture

```
Lunar simulation (Gazebo, 1.62 m/s², μ=1.5)
└── UR10 6-DOF arm
    ├── Hybrid controller: effort control on shoulder_pan/lift + elbow,
    │                      position control on the three wrist joints
    ├── Fault injection  → /fault_label (ground truth) + /degraded_sensor_snapshot
    ├── Performance monitor → energy metrics + CSV logging
    ├── Hybrid FDD node (continuous)   ─┐  alternative
    └── Cascade FDD node (staged)       ┘  detection front-ends
```

### Fault types
| Fault | Injected on | Signature |
|---|---|---|
| bearing_wear | shoulder_lift | friction-scaled effort + velocity noise |
| joint_stiffness | shoulder_pan | effort multiplier + position noise + velocity damping |
| sensor_noise | elbow (+ global IMU) | broadband measurement + IMU noise |

Faults are injected at the signal level (degrading recorded measurements) with
linearly progressing severity; effort is clamped to actuator torque limits.

### Detection pipeline
- **Kalman filter** — constant-velocity, position-only model producing residuals.
- **Isolation Forest** — unsupervised anomaly flag on the fault-relevant feature
  subset (effort + IMU + residual, 78 of 150 features), threshold calibrated to
  10% healthy false-positive rate.
- **SVM** — supervised fault-type classification (4 classes incl. healthy).
- **Random Forest** — fault-severity regression.
- **Cascade** — Layer 1 (statistical screen) → Layer 2 (decision tree) →
  Layer 3 (full hybrid), with asymmetric confidence gating.

## Packages

| Package | Role |
|---|---|
| `construction_robot` | UR10 URDF, controllers, launch, motion scripts |
| `lunar_environment` | Gazebo lunar worlds |
| `Universal_Robots_ROS2_Description` | vendored UR description |
| `fault_injection` | fault injector node + YAML fault configs |
| `performance_monitor` | sensor/energy monitoring |
| `data_logger` | CSV logging library (used by the monitor) |
| `hybrid_fdd` | Kalman/IF/SVM/RF, cascade, training + evaluation scripts |
| `lunar_fdd_interfaces` | custom messages (FaultLabel, FDDResult, FaultStatus, …) |

## Prerequisites
- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic 11
- Python: numpy, scipy, scikit-learn, pandas, joblib, matplotlib, seaborn, psutil

## Build
```bash
git clone https://github.com/tiffaning/lunar_fdd_ws.git
cd lunar_fdd_ws
rosdep install --ignore-src --from-paths src -y
colcon build --symlink-install
source install/setup.bash
```

## Usage

### 1. Launch the robot only
```bash
ros2 launch construction_robot lunar_robot.launch.py
ros2 control list_controllers        # expect arm_effort_controller + wrist_position_controller active
```

### 2. Collect training data (per fault type)
```bash
# terminal 1 — logs degraded sensor data to data/raw-dynamic/
ros2 launch fault_injection fault_experiment.launch.py experiment:=bearing_wear
# terminal 2 — drive the arm through its workspace
ros2 run construction_robot repeated_motion
```
Repeat for `baseline`, `bearing_wear`, `joint_stiffness`, `sensor_noise`.

### 3. Train the models
```bash
python3 src/hybrid_fdd/scripts/train_models.py \
    --data_dir ~/lunar_fdd_ws/data/raw-dynamic \
    --model_dir ~/lunar_fdd_ws/src/hybrid_fdd/models --window_size 100
colcon build --symlink-install && source install/setup.bash
```
Produces `svm_classifier`, `isolation_forest`, `scaler`, `label_encoder`,
`severity_regressor`, `decision_tree`, `anomaly_threshold`, `layer1_baseline`
(`.pkl`) plus a confusion matrix.

### 4. Run detection (choose one front-end)
```bash
# continuous hybrid FDD
ros2 launch hybrid_fdd standard_fdd.launch.py experiment:=bearing_wear
# OR the model cascade
ros2 launch hybrid_fdd cascade_fdd.launch.py experiment:=bearing_wear
# drive the arm (terminal 2)
ros2 run construction_robot repeated_motion
# inspect
ros2 topic echo /fdd_result       # fault_type, confidence, severity, anomaly
ros2 topic echo /fault_status      # cascade layer used + compute time
```

### 5. Phase 5 evaluation (continuous vs. cascade)
```bash
# automated data collection: 5 reps x 4 scenarios x 2 strategies -> data/phase5/
L1=0.9 L2=0.9 L2N=0.7 bash src/hybrid_fdd/scripts/run_phase5.sh 5
# optional threshold grid-search records -> data/phase5_grid/
MODE=grid bash src/hybrid_fdd/scripts/run_phase5.sh
# analysis: metrics, Welch t-tests + ANOVA, plots, Pareto frontier
python3 src/hybrid_fdd/scripts/evaluate_phase5.py \
    --log_dir ~/lunar_fdd_ws/data/phase5 --grid_dir ~/lunar_fdd_ws/data/phase5_grid
```
Outputs `summary.csv`, `significance.csv`, `grid_search.csv`, and figures to
`data/phase5/results/`.

## Repository layout
```
src/
├── construction_robot/       robot description, controllers, motion
├── lunar_environment/        Gazebo worlds
├── fault_injection/          fault injector + configs
├── performance_monitor/      monitoring
├── data_logger/              CSV logging
├── hybrid_fdd/               FDD models, cascade, train/eval scripts
│   ├── hybrid_fdd/           nodes + kalman/feature/classifier modules
│   ├── scripts/              train_models.py, evaluate_phase5.py, run_phase5.sh
│   ├── launch/               standard_fdd, cascade_fdd
│   └── models/               trained .pkl models
├── lunar_fdd_interfaces/     custom messages
└── Universal_Robots_ROS2_Description/
data/
├── raw-dynamic/              training data (degraded sensor CSVs)
├── phase5/                   evaluation logs + results/
└── phase5_grid/              grid-search records
```

## Research Background

This work implements a model cascade for FDD that uses lightweight statistical
screening and a supervised decision tree before escalating to the full hybrid
(Kalman + SVM + severity) system, characterizing the resulting energy–accuracy
trade-off under simulated lunar conditions.

## Author

Tiffani Ng — University of Florida, AiRIS Lab
tiffani.ng@ufl.edu
