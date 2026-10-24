"""
features.py
Derived feature engineering for the cervical neuromuscular stabilization analysis.

All new columns are computed from existing measured columns only.
No physiological variables are invented beyond what is directly derivable.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ROLLING_WINDOW_SAMPLES,
    MOTION_THRESHOLD_DEG_S,
    EMG_MVC_CLIP_PERCENTILE,
)

log = logging.getLogger(__name__)


def compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns to the cleaned DataFrame.

    New columns added (if source columns are present):
    - timestamp_s                : timestamp_ms / 1000
    - sample_index               : row index within each session
    - angle_rad                  : angle_deg in radians
    - sin_angle                  : sin(angle_rad)
    - abs_angular_velocity_deg_s : |angular_velocity_deg_s|
    - emg_percent_mvc_clipped    : %MVC with extreme outliers clipped
    - motion_flag                : 1 if |angular_velocity| > threshold, else 0
    - emg_mvc_rolling_mean       : rolling mean of %MVC (ROLLING_WINDOW_SAMPLES)
    - emg_mvc_rolling_std        : rolling std of %MVC (ROLLING_WINDOW_SAMPLES)

    Parameters
    ----------
    df : cleaned DataFrame (output of cleaning.clean_dataset)

    Returns
    -------
    DataFrame with additional derived columns.
    """
    df = df.copy()

    # ── Timestamp in seconds ──────────────────────────────────────────────────
    if "timestamp_ms" in df.columns:
        df["timestamp_s"] = df["timestamp_ms"] / 1000.0

    # ── Per-session sample index ───────────────────────────────────────────────
    if "session_id" in df.columns:
        df["sample_index"] = df.groupby("session_id").cumcount()
    else:
        df["sample_index"] = np.arange(len(df))

    # ── Angle in radians and sine ─────────────────────────────────────────────
    if "angle_deg" in df.columns:
        df["angle_rad"] = np.deg2rad(df["angle_deg"])
        df["sin_angle"] = np.sin(df["angle_rad"])

    # ── Absolute angular velocity ─────────────────────────────────────────────
    if "angular_velocity_deg_s" in df.columns:
        df["abs_angular_velocity_deg_s"] = df["angular_velocity_deg_s"].abs()

        # Binary motion flag
        df["motion_flag"] = (
            df["abs_angular_velocity_deg_s"] > MOTION_THRESHOLD_DEG_S
        ).astype(int)

    # ── %MVC clipped at high percentile (outlier robustness) ─────────────────
    if "emg_percent_mvc" in df.columns:
        clip_val = df["emg_percent_mvc"].quantile(EMG_MVC_CLIP_PERCENTILE / 100.0)
        df["emg_percent_mvc_clipped"] = df["emg_percent_mvc"].clip(upper=clip_val)
        log.info(
            "  EMG clipped at %.1f percentile (clip value = %.2f %%%%MVC).",
            EMG_MVC_CLIP_PERCENTILE, clip_val,
        )

    # ── Rolling statistics of %MVC ────────────────────────────────────────────
    if "emg_percent_mvc" in df.columns:
        if "session_id" in df.columns:
            grp = df.groupby("session_id")["emg_percent_mvc"]
        else:
            grp = df["emg_percent_mvc"]

        roll_kwargs = dict(window=ROLLING_WINDOW_SAMPLES, min_periods=1)
        if "session_id" in df.columns:
            df["emg_mvc_rolling_mean"] = grp.transform(
                lambda s: s.rolling(**roll_kwargs).mean()
            )
            df["emg_mvc_rolling_std"] = grp.transform(
                lambda s: s.rolling(**roll_kwargs).std()
            )
        else:
            df["emg_mvc_rolling_mean"] = (
                df["emg_percent_mvc"].rolling(**roll_kwargs).mean()
            )
            df["emg_mvc_rolling_std"] = (
                df["emg_percent_mvc"].rolling(**roll_kwargs).std()
            )

    log.info("Feature engineering complete. Columns: %d total.", len(df.columns))
    return df


# ── Optional condition labeling ───────────────────────────────────────────────

def load_conditions_file(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load an optional conditions CSV that maps time ranges to condition labels.

    Expected format:
        session_id, start_ms, end_ms, condition
        session_01, 0,        30000,  baseline
        session_01, 30000,    90000,  static
        session_01, 90000,    150000, dynamic

    Returns the DataFrame, or None if the file does not exist.
    """
    if not filepath.exists():
        log.info("No conditions file found at '%s'. Skipping condition labeling.", filepath)
        return None

    try:
        cond_df = pd.read_csv(filepath, dtype={"session_id": str, "condition": str})
        required = {"session_id", "start_ms", "end_ms", "condition"}
        missing = required - set(cond_df.columns)
        if missing:
            log.warning(
                "Conditions file is missing columns: %s. Skipping.", missing
            )
            return None
        log.info("Loaded conditions file: %d entries.", len(cond_df))
        return cond_df
    except Exception as exc:
        log.warning("Could not read conditions file '%s': %s. Skipping.", filepath, exc)
        return None


def apply_conditions(
    df: pd.DataFrame,
    conditions_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Add a 'condition' column to df by matching each row's session and
    timestamp_ms against the provided time ranges.

    If conditions_df is None or the necessary columns are absent, the
    function adds a 'condition' column filled with 'unlabeled' and returns.

    Parameters
    ----------
    df            : processed DataFrame with session_id and timestamp_ms
    conditions_df : output of load_conditions_file(), or None

    Returns
    -------
    DataFrame with 'condition' column added.
    """
    df = df.copy()

    if conditions_df is None or "timestamp_ms" not in df.columns:
        # Use the firmware-written status column directly (SESSION_TYPE command).
        # Lowercase it so it matches the color keys in plotting.py.
        if "status" in df.columns:
            df["condition"] = df["status"].str.strip().str.lower()
            n_labeled = df["condition"].isin(["static", "dynamic", "baseline", "fatigue", "recovery"]).sum()
            log.info("Condition column mapped from status: %d / %d rows labeled.", n_labeled, len(df))
        else:
            df["condition"] = "unlabeled"
        return df

    df["condition"] = "unlabeled"

    for _, row in conditions_df.iterrows():
        mask = (
            (df["session_id"] == str(row["session_id"]))
            & (df["timestamp_ms"] >= float(row["start_ms"]))
            & (df["timestamp_ms"] <  float(row["end_ms"]))
        )
        df.loc[mask, "condition"] = str(row["condition"])

    n_labeled = (df["condition"] != "unlabeled").sum()
    log.info("Condition labeling: %d / %d rows labeled.", n_labeled, len(df))
    return df
