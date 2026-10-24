"""
config.py
Configuration constants for the cervical neuromuscular stabilization analysis pipeline.

All tunable parameters live here. Edit this file to adjust the analysis without
touching any other module.
"""

# ── CSV Column Definitions ────────────────────────────────────────────────────
# Matches CervicalLogger.ino firmware output.
# angular_accel_deg_s2 was added in the updated firmware; the pipeline handles
# CSV files with or without this column.
RAW_COLUMNS = [
    "timestamp_ms",
    "angle_deg",
    "angular_velocity_deg_s",
    "angular_accel_deg_s2",      # present in updated firmware only
    "cervical_torque_nm",
    "emg_env_raw",
    "emg_raw_raw",
    "emg_rms",
    "emg_percent_mvc",
    "accel_angle_deg",
    "status",
]

NUMERIC_COLUMNS = [
    "timestamp_ms",
    "angle_deg",
    "angular_velocity_deg_s",
    "angular_accel_deg_s2",
    "cervical_torque_nm",
    "emg_env_raw",
    "emg_raw_raw",
    "emg_rms",
    "emg_percent_mvc",
    "accel_angle_deg",
]

# Status field values that indicate valid streaming data.
# Includes legacy "STREAMING" (old firmware) and the four session-type labels
# introduced when SESSION_TYPE command was added to CervicalLogger.ino.
# cleaning.py uppercases the status column before this check.
STREAMING_STATUS_VALUES = {"BASELINE", "STATIC", "DYNAMIC", "FATIGUE", "RECOVERY"}

# ── Analysis Variables ────────────────────────────────────────────────────────
DEPENDENT_VAR = "emg_percent_mvc"

# Model specifications used in regression analysis.
# angular_accel_deg_s2 is intentionally excluded as a predictor per research design.
MODELS = {
    "M1_angle": {
        "features": ["angle_deg"],
        "label": "Model 1: angle_deg only",
    },
    "M2_torque": {
        "features": ["cervical_torque_nm"],
        "label": "Model 2: cervical_torque_nm only",
    },
    "M3_torque_abs_velocity": {
        "features": ["cervical_torque_nm", "abs_angular_velocity_deg_s"],
        "label": "Model 3: torque + |angular velocity|",
    },
}

# ── Feature Engineering ───────────────────────────────────────────────────────
ROLLING_WINDOW_SAMPLES = 20          # 200 ms at 100 Hz
MOTION_THRESHOLD_DEG_S  = 5.0        # |angular velocity| threshold for motion_flag
EMG_MVC_CLIP_PERCENTILE = 99.5       # clip %MVC outliers above this percentile

# ── Data Quality ──────────────────────────────────────────────────────────────
MIN_VALID_ROWS = 30                  # minimum rows needed to run analysis

# ── Segmented Regression ──────────────────────────────────────────────────────
BREAKPOINT_N_CANDIDATES    = 100     # candidate breakpoints to search
BREAKPOINT_PERCENTILE_LOW  = 10      # angle_deg percentile for lower search bound
BREAKPOINT_PERCENTILE_HIGH = 90      # angle_deg percentile for upper search bound
SEGMENT_MIN_POINTS         = 10      # minimum points required in each segment

# ── Cross-Validation ──────────────────────────────────────────────────────────
CV_FOLDS        = 5
CV_RANDOM_STATE = 42

# ── Figures ───────────────────────────────────────────────────────────────────
FIGURE_DPI          = 150
FIGURE_SIZE_DEFAULT = (8, 5)          # inches (width, height)
FIGURE_SIZE_WIDE    = (10, 5)

# Consistent color cycle for up to 4 models
MODEL_COLORS = ["#2166ac", "#d6604d", "#1a9641", "#762a83"]

# ── Default CLI Paths ──────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR  = "data_raw"
DEFAULT_OUTPUT_DIR = "."
