"""
plotting.py
Publication-quality matplotlib figures for the cervical neuromuscular
stabilization analysis.

Style: clean scientific (no seaborn, no unusual colors).
All figures saved as PNG at FIGURE_DPI resolution.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import FIGURE_DPI, FIGURE_SIZE_DEFAULT, FIGURE_SIZE_WIDE, MODEL_COLORS, MODELS

log = logging.getLogger(__name__)

# Apply a clean, readable style
plt.style.use("classic")

_FONT = {"family": "sans-serif", "size": 11}
matplotlib.rc("font", **_FONT)
matplotlib.rc("axes",  titlesize=12, labelsize=11)
matplotlib.rc("xtick", labelsize=10)
matplotlib.rc("ytick", labelsize=10)
matplotlib.rc("legend", fontsize=10)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved figure: %s", path.name)


def _regression_line(x: np.ndarray, y: np.ndarray):
    """Return (x_line, y_line) for a simple OLS regression line."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return np.array([]), np.array([])
    xf, yf = x[finite], y[finite]
    reg = LinearRegression().fit(xf.reshape(-1, 1), yf)
    x_line = np.linspace(xf.min(), xf.max(), 200)
    y_line = reg.predict(x_line.reshape(-1, 1))
    return x_line, y_line


# ── Time-series plots ─────────────────────────────────────────────────────────

def plot_time_series(
    df: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    color: str = "#2166ac",
) -> None:
    """Generic time-series plot: col vs timestamp_s."""
    if col not in df.columns or "timestamp_s" not in df.columns:
        log.warning("Cannot plot '%s': required columns missing.", title)
        return

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    ax.plot(df["timestamp_s"], df[col], color=color, linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, output_path)


# ── Scatter plots with regression line ───────────────────────────────────────

def plot_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    color: str = "#2166ac",
) -> None:
    """Scatter plot of y_col vs x_col with a simple linear regression overlay."""
    if x_col not in df.columns or y_col not in df.columns:
        log.warning("Cannot plot '%s': required columns missing.", title)
        return

    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].values
    y = sub[y_col].values

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    ax.scatter(x, y, color=color, s=6, alpha=0.4, label="Observed")

    x_line, y_line = _regression_line(x, y)
    if len(x_line):
        ax.plot(x_line, y_line, color="#d6604d", linewidth=1.8,
                label="Linear fit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, output_path)


# ── Predicted vs Actual plots ─────────────────────────────────────────────────

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_label: str,
    output_path: Path,
    color: str = "#2166ac",
) -> None:
    """Scatter of predicted vs actual %MVC with a perfect-prediction reference line."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)

    ax.scatter(y_true, y_pred, color=color, s=6, alpha=0.45, label="Predictions")

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, color="#d6604d", linewidth=1.5, linestyle="--",
            label="Perfect prediction (y = x)")

    ax.set_xlabel("Actual EMG %MVC")
    ax.set_ylabel("Predicted EMG %MVC")
    ax.set_title(f"Predicted vs Actual — {model_label}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, output_path)


# ── Segmented regression plot ─────────────────────────────────────────────────

def plot_segmented_regression(
    df: pd.DataFrame,
    seg_results: dict,
    output_path: Path,
) -> None:
    """
    Scatter plot of angle_deg vs emg_percent_mvc with the piecewise linear fit
    and a vertical line at the detected threshold θ*.
    """
    if not seg_results.get("success"):
        log.warning("Segmented regression did not succeed; skipping plot.")
        return

    if "angle_deg" not in df.columns or "emg_percent_mvc" not in df.columns:
        log.warning("Cannot plot segmented regression: required columns missing.")
        return

    sub = df[["angle_deg", "emg_percent_mvc"]].dropna()
    x = sub["angle_deg"].values
    y = sub["emg_percent_mvc"].values

    theta_star  = seg_results["theta_star_deg"]
    intercept   = seg_results["intercept"]
    slope_below = seg_results["slope_below"]
    slope_above = seg_results["slope_above"]

    x_lo = np.linspace(x.min(), theta_star, 150)
    y_lo = intercept + slope_below * x_lo

    # The above-threshold line must be continuous at θ*
    y_at_theta = intercept + slope_below * theta_star
    x_hi = np.linspace(theta_star, x.max(), 150)
    y_hi = y_at_theta + slope_above * (x_hi - theta_star)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_DEFAULT)
    ax.scatter(x, y, color="#aaaaaa", s=6, alpha=0.4, label="Observed", zorder=1)
    ax.plot(x_lo, y_lo, color="#2166ac", linewidth=2.0,
            label=f"Fit below θ* (slope={slope_below:.3f})", zorder=2)
    ax.plot(x_hi, y_hi, color="#d6604d", linewidth=2.0,
            label=f"Fit above θ* (slope={slope_above:.3f})", zorder=2)
    ax.axvline(theta_star, color="#1a9641", linewidth=1.5, linestyle="--",
               label=f"θ* = {theta_star:.1f}°", zorder=3)

    ax.set_xlabel("Neck Flexion Angle (°)")
    ax.set_ylabel("EMG %MVC")
    ax.set_title("Segmented Regression: EMG %MVC vs Neck Flexion Angle")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, output_path)


# ── Condition overlay plot ────────────────────────────────────────────────────

def plot_condition_overlay(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Time-series of EMG %MVC with colored background regions per condition.
    Only generated when a 'condition' column is present and has >1 unique label.
    """
    if "condition" not in df.columns or "timestamp_s" not in df.columns:
        log.info("Condition overlay skipped: 'condition' or 'timestamp_s' missing.")
        return

    unique_conds = df["condition"].unique()
    if len(unique_conds) <= 1 and unique_conds[0] == "unlabeled":
        log.info("Condition overlay skipped: all rows are 'unlabeled'.")
        return

    cond_colors = {
        "baseline": "#b2df8a",
        "static":   "#a6cee3",
        "dynamic":  "#fdbf6f",
        "unlabeled":"#f0f0f0",
    }

    cond_line_colors = {
        "static":   "#2166ac",
        "dynamic":  "#d95f02",
        "baseline": "#33a02c",
        "unlabeled":"#aaaaaa",
    }

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    if "emg_percent_mvc" in df.columns:
        for cond in ["static", "dynamic", "baseline"]:
            if cond not in df["condition"].values:
                continue
            # Mask out other conditions with NaN so gaps appear naturally
            y = df["emg_percent_mvc"].copy().astype(float)
            y[df["condition"] != cond] = np.nan
            ax.plot(df["timestamp_s"], y,
                    color=cond_line_colors[cond], linewidth=0.8, alpha=0.9,
                    label=cond.capitalize(), zorder=3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG %MVC")
    ax.set_title("EMG %MVC Over Time — STATIC (blue) vs DYNAMIC (orange)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, output_path)


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_plots(
    df: pd.DataFrame,
    model_predictions: Dict,
    seg_results: dict,
    figures_dir: Path,
) -> None:
    """
    Generate and save all required figures.

    Parameters
    ----------
    df                : processed DataFrame
    model_predictions : output of modeling.get_model_predictions()
    seg_results       : output of modeling.run_segmented_regression()[0]
    figures_dir       : directory where PNG files are saved
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = MODEL_COLORS

    # 1. Neck flexion angle over time
    plot_time_series(
        df, "angle_deg", "Neck Flexion Angle (°)",
        "Cervical Flexion Angle Over Time",
        figures_dir / "angle_vs_time.png",
        color=colors[0],
    )

    # 2. Angular velocity over time
    plot_time_series(
        df, "angular_velocity_deg_s", "Angular Velocity (°/s)",
        "Cervical Angular Velocity Over Time",
        figures_dir / "velocity_vs_time.png",
        color=colors[1],
    )

    # 3. Cervical torque over time
    plot_time_series(
        df, "cervical_torque_nm", "Cervical Torque (Nm)",
        "Estimated Cervical Torque Over Time",
        figures_dir / "torque_vs_time.png",
        color=colors[2],
    )

    # 4. EMG %MVC over time
    plot_time_series(
        df, "emg_percent_mvc", "EMG %MVC",
        "Upper Trapezius EMG Activation (%MVC) Over Time",
        figures_dir / "emg_percent_mvc_vs_time.png",
        color=colors[3],
    )

    # 5. EMG vs angle scatter
    plot_scatter_with_regression(
        df, "angle_deg", "emg_percent_mvc",
        "Neck Flexion Angle (°)", "EMG %MVC",
        "EMG %MVC vs Neck Flexion Angle",
        figures_dir / "emg_vs_angle_scatter.png",
        color=colors[0],
    )

    # 6. EMG vs torque scatter
    plot_scatter_with_regression(
        df, "cervical_torque_nm", "emg_percent_mvc",
        "Cervical Torque (Nm)", "EMG %MVC",
        "EMG %MVC vs Cervical Torque",
        figures_dir / "emg_vs_torque_scatter.png",
        color=colors[1],
    )

    # 7. EMG vs angular velocity scatter
    plot_scatter_with_regression(
        df, "angular_velocity_deg_s", "emg_percent_mvc",
        "Angular Velocity (°/s)", "EMG %MVC",
        "EMG %MVC vs Angular Velocity",
        figures_dir / "emg_vs_velocity_scatter.png",
        color=colors[2],
    )

    # 8. Predicted vs actual — static model (M2)
    if "M2_torque" in model_predictions:
        pred = model_predictions["M2_torque"]
        plot_predicted_vs_actual(
            pred["y_true"], pred["y_pred"],
            MODELS["M2_torque"]["label"],
            figures_dir / "predicted_vs_actual_static.png",
            color=colors[1],
        )

    # 9. Predicted vs actual — dynamic model (M3: torque + abs velocity)
    for mid in ("M3_torque_abs_velocity",):
        if mid in model_predictions:
            pred = model_predictions[mid]
            plot_predicted_vs_actual(
                pred["y_true"], pred["y_pred"],
                MODELS[mid]["label"],
                figures_dir / "predicted_vs_actual_dynamic.png",
                color=colors[2],
            )
            break

    # 10. Segmented regression threshold plot
    plot_segmented_regression(
        df, seg_results,
        figures_dir / "segmented_threshold_plot.png",
    )

    # 11. Condition overlay (only if labeling was applied)
    plot_condition_overlay(
        df,
        figures_dir / "optional_condition_overlay.png",
    )
