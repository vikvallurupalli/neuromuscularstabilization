"""
cleaning.py
Data cleaning pipeline for raw CervicalLogger CSV data.

Cleaning steps applied in order:
  1. Coerce all numeric columns to float (drop non-parseable rows)
  2. Filter to STREAMING rows only
  3. Drop rows where emg_percent_mvc <= 0 (uncalibrated, firmware outputs -1)
  4. Sort by timestamp within each session
  5. Reset sample index

A data quality report is returned alongside the cleaned DataFrame.
"""

import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

from config import NUMERIC_COLUMNS, STREAMING_STATUS_VALUES, MIN_VALID_ROWS

log = logging.getLogger(__name__)


def clean_dataset(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the full cleaning pipeline on the raw combined DataFrame.

    Parameters
    ----------
    raw_df : combined raw DataFrame from io_utils.load_all_sessions()

    Returns
    -------
    (cleaned_df, quality_report_dict)
    """
    n_input = len(raw_df)
    drop_counts: Dict[str, int] = {}

    df = raw_df.copy()

    # ── Step 1: coerce numerics ───────────────────────────────────────────────
    df, n_dropped_numeric = _coerce_numerics(df)
    drop_counts["non_numeric"] = n_dropped_numeric

    # ── Step 2: filter to STREAMING rows ─────────────────────────────────────
    if "status" in df.columns:
        before = len(df)
        df["status"] = df["status"].astype(str).str.strip().str.upper()
        df = df[df["status"].isin(STREAMING_STATUS_VALUES)].copy()
        n_dropped_status = before - len(df)
        drop_counts["non_streaming_status"] = n_dropped_status
        log.info("  Status filter: dropped %d non-STREAMING rows.", n_dropped_status)
    else:
        log.warning("  'status' column not found; skipping status filter.")
        drop_counts["non_streaming_status"] = 0

    # ── Step 3: drop rows where emg_rms <= 0 (sensor not active) ────────────
    if "emg_rms" in df.columns:
        before = len(df)
        df = df[df["emg_rms"] > 0.0].copy()
        n_dropped_mvc = before - len(df)
        drop_counts["uncalibrated_mvc"] = n_dropped_mvc
        log.info("  EMG RMS filter: dropped %d zero/negative rows.", n_dropped_mvc)
    else:
        drop_counts["uncalibrated_mvc"] = 0

    # ── Step 4: drop exact duplicate rows ─────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates().copy()
    n_dropped_dup = before - len(df)
    drop_counts["duplicates"] = n_dropped_dup
    if n_dropped_dup:
        log.info("  Duplicates removed: %d.", n_dropped_dup)

    # ── Step 5: sort by session then timestamp ────────────────────────────────
    sort_cols = []
    if "session_id" in df.columns:
        sort_cols.append("session_id")
    if "timestamp_ms" in df.columns:
        sort_cols.append("timestamp_ms")
    if sort_cols:
        df = df.sort_values(sort_cols).copy()

    df = df.reset_index(drop=True)

    n_valid = len(df)
    n_dropped_total = n_input - n_valid

    quality_report = {
        "n_input_rows":   n_input,
        "n_valid_rows":   n_valid,
        "n_dropped_total": n_dropped_total,
        "drop_breakdown": drop_counts,
        "sessions": (
            df["session_id"].nunique() if "session_id" in df.columns else 1
        ),
        "columns_present": list(df.columns),
    }

    log.info(
        "Cleaning complete: %d → %d valid rows (%d dropped).",
        n_input, n_valid, n_dropped_total,
    )

    if n_valid < MIN_VALID_ROWS:
        log.warning(
            "Only %d valid rows remain after cleaning (minimum recommended: %d). "
            "Analysis results may be unreliable.",
            n_valid, MIN_VALID_ROWS,
        )

    return df, quality_report


# ── Internal helpers ──────────────────────────────────────────────────────────

def _coerce_numerics(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Convert numeric columns to float64. Rows where any required numeric column
    cannot be parsed are dropped.

    Columns listed in NUMERIC_COLUMNS that are absent from the DataFrame are
    silently skipped (supports both old and new firmware CSV formats).
    """
    present_numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
    before = len(df)

    for col in present_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that have NaN in any numeric column that is present
    df = df.dropna(subset=present_numeric).copy()
    n_dropped = before - len(df)
    if n_dropped:
        log.info("  Numeric coercion: dropped %d unparseable rows.", n_dropped)
    return df, n_dropped
