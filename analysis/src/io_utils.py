"""
io_utils.py
Utilities for reading raw CSV files produced by the CervicalLogger firmware.

The firmware outputs a mix of:
  - Comment lines starting with '#'
  - A CSV header line (may appear more than once due to START commands)
  - Data rows

This module locates the first valid header, skips all junk before it,
and returns clean DataFrames with a session_id column.
"""

import logging
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import pandas as pd

log = logging.getLogger(__name__)

# The token that uniquely identifies a valid CSV header line
_HEADER_TOKEN = "timestamp_ms"


def find_csv_files(input_dir: Path) -> List[Path]:
    """Return all .csv files in input_dir sorted alphabetically."""
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No .csv files found in '{input_dir}'. "
            "Place raw data files there before running the pipeline."
        )
    log.info("Found %d CSV file(s) in '%s'.", len(files), input_dir)
    return files


def read_raw_file(filepath: Path, session_id: str) -> Tuple[pd.DataFrame, dict]:
    """
    Read one raw CSV file from the CervicalLogger firmware.

    Robustly handles:
    - Startup comment lines (prefixed with '#')
    - Arbitrary junk before the CSV header
    - Repeated CSV header lines (e.g., from multiple START commands)
    - Lines with wrong number of fields (skipped)

    Parameters
    ----------
    filepath   : path to the .csv file
    session_id : label to attach to every row (typically the filename stem)

    Returns
    -------
    (df, stats) where stats is a dict with row-count diagnostics.
    """
    try:
        raw_text = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("Cannot read file '%s': %s", filepath, exc)
        return pd.DataFrame(), _empty_stats(filepath.name, reason=str(exc))

    lines = raw_text.splitlines()

    # Locate the first header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith(_HEADER_TOKEN):
            header_idx = i
            break

    if header_idx is None:
        log.warning("No CSV header found in '%s' — skipping.", filepath.name)
        return pd.DataFrame(), _empty_stats(filepath.name, reason="no_header_found")

    header_line = lines[header_idx].strip()

    # Collect data lines: skip comments, empty lines, repeated headers
    data_lines: List[str] = []
    total_candidate = 0
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith(_HEADER_TOKEN):
            continue  # repeated header from a re-start
        total_candidate += 1
        data_lines.append(stripped)

    if not data_lines:
        log.warning("No data rows in '%s' after header.", filepath.name)
        return pd.DataFrame(), _empty_stats(filepath.name, reason="no_data_rows",
                                            raw_rows=total_candidate)

    # Parse with pandas (bad lines silently dropped)
    csv_blob = header_line + "\n" + "\n".join(data_lines)
    try:
        df = pd.read_csv(StringIO(csv_blob), dtype=str, on_bad_lines="skip")
    except Exception as exc:
        log.error("pandas failed to parse '%s': %s", filepath.name, exc)
        return pd.DataFrame(), _empty_stats(filepath.name, reason=str(exc),
                                            raw_rows=total_candidate)

    df["session_id"] = session_id

    stats = {
        "file":        filepath.name,
        "session_id":  session_id,
        "raw_rows":    total_candidate,
        "parsed_rows": len(df),
        "reason":      "ok",
    }
    log.info("  '%s' → %d candidate rows, %d parsed.", filepath.name,
             total_candidate, len(df))
    return df, stats


def load_all_sessions(input_dir: Path) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Load every CSV file in input_dir and concatenate into one DataFrame.

    Parameters
    ----------
    input_dir : directory containing raw .csv files

    Returns
    -------
    (combined_df, list_of_per_file_stats)
    """
    files = find_csv_files(input_dir)
    frames: List[pd.DataFrame] = []
    all_stats: List[dict] = []

    for fpath in files:
        session_id = fpath.stem
        df, stats = read_raw_file(fpath, session_id)
        all_stats.append(stats)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(
            "No valid data could be loaded from any file in "
            f"'{input_dir}'. Check that the files contain firmware CSV output."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d total rows from %d file(s).", len(combined), len(frames))
    return combined, all_stats


# ── Internal helpers ──────────────────────────────────────────────────────────

def _empty_stats(filename: str, reason: str = "", raw_rows: int = 0) -> dict:
    return {
        "file":        filename,
        "session_id":  None,
        "raw_rows":    raw_rows,
        "parsed_rows": 0,
        "reason":      reason or "unknown",
    }
