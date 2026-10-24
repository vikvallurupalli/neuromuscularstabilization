"""
modeling.py
Regression modeling and change-point analysis for the cervical neuromuscular
stabilization project.

Models fitted:
  M1: emg_percent_mvc ~ angle_deg
  M2: emg_percent_mvc ~ cervical_torque_nm
  M3: emg_percent_mvc ~ cervical_torque_nm + angular_velocity_deg_s
  M4: emg_percent_mvc ~ cervical_torque_nm + abs_angular_velocity_deg_s

Also fits standardized-coefficient versions of all models and runs
k-fold cross-validation for predictive error estimates.

Segmented regression searches for a breakpoint angle θ* that minimizes
residual error in a piecewise linear fit of emg_percent_mvc ~ angle_deg.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from config import (
    BREAKPOINT_N_CANDIDATES,
    BREAKPOINT_PERCENTILE_HIGH,
    BREAKPOINT_PERCENTILE_LOW,
    CV_FOLDS,
    CV_RANDOM_STATE,
    DEPENDENT_VAR,
    MIN_VALID_ROWS,
    MODELS,
    SEGMENT_MIN_POINTS,
)

log = logging.getLogger(__name__)


# ── Public entry points ───────────────────────────────────────────────────────

def run_all_models(
    df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit all regression models defined in config.MODELS.

    Returns
    -------
    ols_results   : {model_id: statsmodels RegressionResultsWrapper}
    comparison_df : DataFrame with one row per model and key metrics
    coeff_df      : DataFrame with all coefficient estimates
    cv_df         : DataFrame with cross-validation results
    """
    comparison_rows: List[dict] = []
    coeff_rows:      List[dict] = []
    cv_rows:         List[dict] = []
    ols_results:     Dict[str, Any] = {}

    for model_id, spec in MODELS.items():
        features = spec["features"]
        label    = spec["label"]

        # Check all required columns exist
        missing = [f for f in features + [DEPENDENT_VAR] if f not in df.columns]
        if missing:
            log.warning("Model '%s' skipped: missing columns %s.", model_id, missing)
            continue

        sub = df[features + [DEPENDENT_VAR]].dropna()
        if len(sub) < MIN_VALID_ROWS:
            log.warning(
                "Model '%s' skipped: only %d valid rows (need %d).",
                model_id, len(sub), MIN_VALID_ROWS,
            )
            continue

        X = sub[features].values
        y = sub[DEPENDENT_VAR].values

        # ── statsmodels OLS ───────────────────────────────────────────────────
        X_sm = sm.add_constant(X, has_constant="add")
        ols  = sm.OLS(y, X_sm).fit()
        ols_results[model_id] = ols

        y_pred_ols = ols.fittedvalues
        rmse_ols   = float(np.sqrt(mean_squared_error(y, y_pred_ols)))
        mae_ols    = float(mean_absolute_error(y, y_pred_ols))

        comparison_rows.append({
            "model_id":        model_id,
            "label":           label,
            "n_features":      len(features),
            "n_obs":           len(sub),
            "r_squared":       round(ols.rsquared, 4),
            "adj_r_squared":   round(ols.rsquared_adj, 4),
            "rmse":            round(rmse_ols, 4),
            "mae":             round(mae_ols, 4),
            "aic":             round(ols.aic, 2),
        })

        # Coefficient table — np.asarray() handles both ndarray and Series
        col_names  = ["const"] + features
        params_v   = np.asarray(ols.params)
        bse_v      = np.asarray(ols.bse)
        tvalues_v  = np.asarray(ols.tvalues)
        pvalues_v  = np.asarray(ols.pvalues)
        ci_v       = np.asarray(ols.conf_int())   # shape (n_params, 2)
        for i, cname in enumerate(col_names):
            coeff_rows.append({
                "model_id":  model_id,
                "term":      cname,
                "coef":      _safe_round(params_v[i],  6),
                "std_err":   _safe_round(bse_v[i],     6),
                "t_value":   _safe_round(tvalues_v[i], 4),
                "p_value":   _safe_round(pvalues_v[i], 6),
                "ci_lower":  _safe_round(ci_v[i, 0],   6),
                "ci_upper":  _safe_round(ci_v[i, 1],   6),
            })

        # ── Standardized coefficients ─────────────────────────────────────────
        std_coefs = _standardized_coefficients(X, y, features)
        for cname, val in std_coefs.items():
            coeff_rows.append({
                "model_id":  model_id + "_std",
                "term":      cname,
                "coef":      _safe_round(val, 6),
                "std_err":   np.nan,
                "t_value":   np.nan,
                "p_value":   np.nan,
                "ci_lower":  np.nan,
                "ci_upper":  np.nan,
            })

        # ── Scikit-learn cross-validation ─────────────────────────────────────
        cv_rmse, cv_r2 = _run_cv(X, y)
        cv_rows.append({
            "model_id":      model_id,
            "label":         label,
            "cv_folds":      CV_FOLDS,
            "cv_rmse_mean":  round(cv_rmse.mean(), 4),
            "cv_rmse_std":   round(cv_rmse.std(),  4),
            "cv_r2_mean":    round(cv_r2.mean(),   4),
            "cv_r2_std":     round(cv_r2.std(),    4),
        })

        # Add CV results to comparison table
        comparison_rows[-1]["cv_rmse_mean"] = round(cv_rmse.mean(), 4)
        comparison_rows[-1]["cv_r2_mean"]   = round(cv_r2.mean(),   4)

        log.info(
            "  %s | R²=%.3f | RMSE=%.3f | CV-RMSE=%.3f",
            model_id, ols.rsquared, rmse_ols, cv_rmse.mean(),
        )

    comparison_df = pd.DataFrame(comparison_rows)
    coeff_df      = pd.DataFrame(coeff_rows)
    cv_df         = pd.DataFrame(cv_rows)

    return ols_results, comparison_df, coeff_df, cv_df


def run_segmented_regression(
    df: pd.DataFrame,
    angle_col:  str = "angle_deg",
    target_col: str = DEPENDENT_VAR,
) -> Tuple[dict, pd.DataFrame]:
    """
    Piecewise linear (segmented) regression to detect a flexion threshold θ*.

    Method:
      For each candidate breakpoint θ_b in the search range:
        1. Construct hinge feature: h = max(0, angle - θ_b)
        2. Fit OLS: target ~ angle + h  (= two-line model joined at θ_b)
        3. Compute residual sum of squares (RSS)
      Choose θ_b that minimises RSS.

    Slope interpretation:
      - slope_below = β_angle
      - slope_above = β_angle + β_hinge

    Parameters
    ----------
    df         : processed DataFrame
    angle_col  : predictor column name
    target_col : dependent variable column name

    Returns
    -------
    (results_dict, search_df) where search_df has one row per candidate breakpoint.
    """
    sub = df[[angle_col, target_col]].dropna()

    if len(sub) < 2 * SEGMENT_MIN_POINTS + 10:
        msg = (
            f"Insufficient data for segmented regression "
            f"({len(sub)} rows; need at least {2 * SEGMENT_MIN_POINTS + 10})."
        )
        log.warning(msg)
        return {"success": False, "reason": msg}, pd.DataFrame()

    x = sub[angle_col].values
    y = sub[target_col].values

    lo = float(np.percentile(x, BREAKPOINT_PERCENTILE_LOW))
    hi = float(np.percentile(x, BREAKPOINT_PERCENTILE_HIGH))

    if lo >= hi:
        msg = (
            f"Breakpoint search range is degenerate "
            f"(p{BREAKPOINT_PERCENTILE_LOW}={lo:.2f}, "
            f"p{BREAKPOINT_PERCENTILE_HIGH}={hi:.2f}). "
            "Ensure the dataset covers a range of neck angles."
        )
        log.warning(msg)
        return {"success": False, "reason": msg}, pd.DataFrame()

    candidates = np.linspace(lo, hi, BREAKPOINT_N_CANDIDATES)
    search_rows: List[dict] = []

    best_rss      = np.inf
    best_theta    = np.nan
    best_beta     = None

    for theta_b in candidates:
        hinge = np.maximum(0.0, x - theta_b)
        X_seg = np.column_stack([np.ones_like(x), x, hinge])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X_seg, y, rcond=None)
            y_fit = X_seg @ beta
            rss = float(np.sum((y - y_fit) ** 2))
        except np.linalg.LinAlgError:
            rss = np.inf
            beta = np.array([np.nan, np.nan, np.nan])

        search_rows.append({"theta_b": round(theta_b, 4), "rss": round(rss, 4)})

        if rss < best_rss:
            best_rss   = rss
            best_theta = theta_b
            best_beta  = beta

    search_df = pd.DataFrame(search_rows)

    if best_beta is None or np.any(np.isnan(best_beta)):
        msg = "Segmented regression optimisation produced only NaN results."
        log.warning(msg)
        return {"success": False, "reason": msg}, search_df

    intercept    = float(best_beta[0])
    slope_below  = float(best_beta[1])
    slope_above  = float(best_beta[1] + best_beta[2])

    # Count points in each segment
    n_below = int(np.sum(x <= best_theta))
    n_above = int(np.sum(x >  best_theta))

    if n_below < SEGMENT_MIN_POINTS or n_above < SEGMENT_MIN_POINTS:
        msg = (
            f"Best breakpoint θ*={best_theta:.2f}° leaves a segment with "
            f"fewer than {SEGMENT_MIN_POINTS} points "
            f"(below: {n_below}, above: {n_above}). "
            "Threshold estimate is unreliable."
        )
        log.warning(msg)

    results = {
        "success":      True,
        "theta_star_deg": round(float(best_theta), 3),
        "slope_below":  round(slope_below, 5),
        "slope_above":  round(slope_above, 5),
        "intercept":    round(intercept, 5),
        "best_rss":     round(best_rss, 4),
        "n_below":      n_below,
        "n_above":      n_above,
        "n_total":      len(sub),
    }

    log.info(
        "  Segmented regression: θ*=%.2f° | slope_below=%.4f | slope_above=%.4f",
        best_theta, slope_below, slope_above,
    )
    return results, search_df


def run_models_by_condition(
    df: pd.DataFrame,
    condition_col: str = "status",
    target_conditions: tuple = ("STATIC", "DYNAMIC"),
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run all regression models separately for each experimental condition.

    Uses the firmware-written 'status' column (set by SESSION_TYPE command)
    to split data, then calls run_all_models() on each subset independently.

    This is the primary test of H2: compare M2 (torque only) vs M3/M4
    (torque + ω) within the DYNAMIC condition, where ω actually varies.

    Returns
    -------
    per_condition : {label: {"comparison_df", "coeff_df", "cv_df", "ols_results"}}
    combined_df   : all per-condition comparison rows with a leading 'condition' column
    """
    if condition_col not in df.columns:
        log.warning("Column '%s' not found; cannot split by condition.", condition_col)
        return {}, pd.DataFrame()

    per_condition: Dict[str, Any] = {}
    combined_rows: List[pd.DataFrame] = []

    for cond in target_conditions:
        subset = df[df[condition_col] == cond]
        n = len(subset)
        if n < MIN_VALID_ROWS:
            log.warning(
                "Condition '%s': only %d rows (need %d). Skipping.",
                cond, n, MIN_VALID_ROWS,
            )
            continue

        log.info("  Condition '%s': %d rows — running models...", cond, n)
        ols, comparison, coeffs, cv = run_all_models(subset)

        if not comparison.empty:
            comparison.insert(0, "condition", cond)
        per_condition[cond] = {
            "ols_results":   ols,
            "comparison_df": comparison,
            "coeff_df":      coeffs,
            "cv_df":         cv,
        }
        combined_rows.append(comparison)

    combined_df = (
        pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()
    )

    if combined_df.empty:
        log.warning("No conditions had sufficient data. Check that STATUS column contains STATIC/DYNAMIC rows.")
    else:
        log.info("  Per-condition analysis complete. Conditions: %s", list(per_condition.keys()))

    return per_condition, combined_df


def get_model_predictions(
    ols_results: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Return fitted values (in-sample predictions) for each OLS model.

    Parameters
    ----------
    ols_results : output of run_all_models()[0]
    df          : processed DataFrame

    Returns
    -------
    {model_id: array_of_predicted_values}
    """
    predictions = {}
    for model_id, spec in MODELS.items():
        if model_id not in ols_results:
            continue
        features = spec["features"]
        sub = df[features + [DEPENDENT_VAR]].dropna()
        X_sm = sm.add_constant(sub[features].values, has_constant="add")
        predictions[model_id] = {
            "y_pred": ols_results[model_id].predict(X_sm),
            "y_true": sub[DEPENDENT_VAR].values,
        }
    return predictions


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run_cv(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-fold cross-validation and return arrays of RMSE and R² scores."""
    kf  = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    reg = LinearRegression()

    neg_mse = cross_val_score(reg, X, y, cv=kf, scoring="neg_mean_squared_error")
    r2      = cross_val_score(reg, X, y, cv=kf, scoring="r2")

    rmse = np.sqrt(-neg_mse)
    return rmse, r2


def _safe_round(val, ndigits: int = 6):
    """Convert val to float and round; return None if NaN or unconvertible."""
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, ndigits)
    except (TypeError, ValueError):
        return None


def _standardized_coefficients(
    X: np.ndarray, y: np.ndarray, feature_names: List[str]
) -> Dict[str, float]:
    """Return standardized (beta) coefficients by z-scoring X and y before OLS."""
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_z = scaler_X.fit_transform(X)
    y_z = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_sm = sm.add_constant(X_z, has_constant="add")
    try:
        ols = sm.OLS(y_z, X_sm).fit()
        return {
            name: float(ols.params[i + 1])
            for i, name in enumerate(feature_names)
        }
    except Exception as exc:
        log.warning("Standardized regression failed: %s", exc)
        return {name: np.nan for name in feature_names}
