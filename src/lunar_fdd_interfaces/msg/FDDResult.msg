# Published by hybrid FDD node
# Consumed by: Phase 4 cascade controller, Phase 5 evaluation
string fault_type               # detected fault type
string affected_joint           # which joint
float64 confidence              # 0.0-1.0 detection confidence
float64 severity_estimate       # estimated severity level
bool anomaly_detected           # did Kalman filter flag anomaly?
float64 max_residual            # largest Kalman residual
float64 processing_time_ms      # computation time for Phase 5 comparison
string detection_mode           # "continuous" or "cascade" for Phase 5
builtin_interfaces/Time timestamp