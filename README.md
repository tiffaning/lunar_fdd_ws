# Lunar Construction Robot Fault Detection System

Hybrid fault detection and diagnosis system for energy-efficient lunar construction robots using model cascades.

## Project Status

- **Phase 1**: ✅ COMPLETE - Robot simulation environment
- **Phase 2**: 🚧 IN PROGRESS - Performance monitoring + fault injection
- **Phase 3**: 📋 PLANNED - Hybrid FDD algorithms
- **Phase 4**: 📋 PLANNED - Model cascade implementation
- **Phase 5**: 📋 PLANNED - Testing and evaluation

## System Architecture
Robot Simulation Environment
├── Lunar Environment (gravity, regolith physics)
├── UR10 Robot Model (6-DOF construction arm)
├── ROS2 Control System (joint controllers, sensors)
├── Fault Injection System (progressive degradation)
├── Hybrid FDD System (physics + ML models)
├── Model Cascade Logic (energy-efficient detection)
└── Performance Monitor (computational metrics)


## Quick Start

### Prerequisites
- Ubuntu 22.04
- ROS2 Humble
- Gazebo Classic 11

### Build and Run
```bash
# Clone repository
git clone https://github.com/tiffaning/lunar_fdd_ws.git
cd lunar_fdd_ws

# Install dependencies
rosdep install --ignore-src --from-paths src -y

# Build workspace
colcon build --symlink-install
source install/setup.bash

# Launch simulation
ros2 launch construction_robot lunar_robot.launch.py

# Test robot movement (in new terminal)
ros2 run construction_robot safe_joint_test

# Verification
# Check sensor data
ros2 topic hz /joint_states
ros2 topic hz /lunar_robot/imu

# Check controllers
ros2 control list_controllers

Research Background
This system addresses the energy-computational trade-off in fault detection for lunar construction robots by implementing model cascades that use lightweight anomaly detection before escalating to full hybrid FDD processing.

Author
Tiffani Ng - University of Florida
