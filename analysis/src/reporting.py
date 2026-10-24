"""
reporting.py
Exports all analysis artifacts: tables (CSV), Markdown report, and JSON summaries.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import DEPENDENT_VAR, MODELS

log = logging.getLogger(__name__)


# ── Table exports ─────────────────────────────────────────────────────────────

def save_tables(
    df:                      pd.DataFrame,
    comparison_df:           pd.DataFrame,
    coeff_df:                pd.DataFrame,
    cv_df:                   pd.DataFrame,
    seg_results:             dict,
    tables_dir:              Path,
    condition_comparison_df: pd.DataFrame = None,
) -> None:
    """Save all analysis tables as CSV files in tables_dir."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].describe().T
    summary.to_csv(tables_dir / "summary_statistics.csv")
    log.info("  Saved: summary_statistics.csv")

    # Model comparison
    if not comparison_df.empty:
        comparison_df.to_csv(tables_dir / "model_comparison.csv", index=False)
        log.info("  Saved: model_comparison.csv")

    # Regression coefficients
    if not coeff_df.empty:
        coeff_df.to_csv(tables_dir / "regression_coefficients.csv", index=False)
        log.info("  Saved: regression_coefficients.csv")

    # Cross-validation results
    if not cv_df.empty:
        cv_df.to_csv(tables_dir / "cross_validation_results.csv", index=False)
        log.info("  Saved: cross_validation_results.csv")

    # Segmented regression results
    seg_export = {k: v for k, v in seg_results.items()
                  if isinstance(v, (int, float, bool, str))}
    seg_df = pd.DataFrame([seg_export])
    seg_df.to_csv(tables_dir / "segmented_regression_results.csv", index=False)
    log.info("  Saved: segmented_regression_results.csv")

    # Per-condition model comparison (Static vs Dynamic)
    if condition_comparison_df is not None and not condition_comparison_df.empty:
        condition_comparison_df.to_csv(
            tables_dir / "model_comparison_by_condition.csv", index=False
        )
        log.info("  Saved: model_comparison_by_condition.csv")


# ── Quality report ────────────────────────────────────────────────────────────

def write_quality_report(
    quality_report: dict,
    file_stats:     List[dict],
    reports_dir:    Path,
) -> None:
    """Save the data quality report as JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at":  datetime.now().isoformat(),
        "overall":       quality_report,
        "per_file":      file_stats,
    }
    path = reports_dir / "data_quality_report.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    log.info("  Saved: data_quality_report.json")


# ── JSON summary ──────────────────────────────────────────────────────────────

def write_json_summary(
    df:             pd.DataFrame,
    comparison_df:  pd.DataFrame,
    seg_results:    dict,
    quality_report: dict,
    reports_dir:    Path,
) -> None:
    """Save a machine-readable summary of key metrics as JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_model = _best_model(comparison_df)

    summary = {
        "generated_at":   datetime.now().isoformat(),
        "n_valid_rows":   quality_report.get("n_valid_rows"),
        "n_sessions":     quality_report.get("sessions"),
        "dependent_var":  DEPENDENT_VAR,
        "best_model": best_model,
        "segmented_regression": {
            k: v for k, v in seg_results.items()
            if isinstance(v, (int, float, bool, str))
        },
        "model_metrics": (
            comparison_df.to_dict(orient="records")
            if not comparison_df.empty else []
        ),
    }

    path = reports_dir / "analysis_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default))
    log.info("  Saved: analysis_summary.json")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_markdown_report(
    df:                      pd.DataFrame,
    comparison_df:           pd.DataFrame,
    seg_results:             dict,
    quality_report:          dict,
    file_stats:              List[dict],
    reports_dir:             Path,
    condition_comparison_df: pd.DataFrame = None,
) -> None:
    """
    Generate a plain-language Markdown summary report.
    Language is cautious and scientific; no causal claims beyond what
    regression results support.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    a = lines.append   # shorthand

    a("# Cervical Neuromuscular Stabilization — Analysis Report")
    a("")
    a(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    a("")

    # ── Input files ───────────────────────────────────────────────────────────
    a("## 1. Input Data")
    a("")
    a(f"- Files processed: **{len(file_stats)}**")
    for s in file_stats:
        a(f"  - `{s['file']}` — {s.get('parsed_rows', 0)} rows parsed")
    a(f"- Total valid rows after cleaning: **{quality_report.get('n_valid_rows', 'N/A')}**")
    a(f"- Sessions: **{quality_report.get('sessions', 'N/A')}**")
    a("")

    # ── Descriptive statistics ────────────────────────────────────────────────
    a("## 2. Descriptive Statistics")
    a("")
    key_cols = ["angle_deg", "angular_velocity_deg_s", "cervical_torque_nm",
                "emg_percent_mvc"]
    key_cols = [c for c in key_cols if c in df.columns]

    if key_cols:
        desc = df[key_cols].describe().round(3)
        a(desc.to_markdown())
    a("")

    # ── Regression models ─────────────────────────────────────────────────────
    a("## 3. Regression Model Results")
    a("")
    if not comparison_df.empty:
        display_cols = [c for c in
                        ["model_id", "label", "n_obs", "r_squared",
                         "adj_r_squared", "rmse", "aic", "cv_rmse_mean"]
                        if c in comparison_df.columns]
        a(comparison_df[display_cols].round(4).to_markdown(index=False))
    else:
        a("*No models could be fitted (insufficient data).*")
    a("")

    # ── H1 test: nonlinearity — M1 (linear angle) vs M2 (torque = sin(θ)) ──────
    a("## 4. H1 Test — Nonlinearity of EMG vs Angle")
    a("")
    a("Since cervical torque = 50 N × 0.08 m × sin(θ), Model M2 fits EMG ~ sin(θ), "
      "which is the nonlinear function of angle predicted by H1. "
      "Comparing M1 (linear angle) to M2 (torque ∝ sin θ) directly tests whether "
      "EMG is better predicted by a nonlinear transformation of angle.")
    a("")
    if not comparison_df.empty:
        m1 = comparison_df[comparison_df["model_id"] == "M1_angle"]
        m2 = comparison_df[comparison_df["model_id"] == "M2_torque"]
        if not m1.empty and not m2.empty:
            m1_r2 = float(m1["r_squared"].values[0] or 0)
            m2_r2 = float(m2["r_squared"].values[0] or 0)
            if m2_r2 > m1_r2:
                a(f"**Result:** M2 (R² = {m2_r2:.3f}) > M1 (R² = {m1_r2:.3f}) — "
                  f"supports H1. The sin(θ) model explains more variance than linear angle.")
            else:
                a(f"**Result:** M1 (R² = {m1_r2:.3f}) ≥ M2 (R² = {m2_r2:.3f}) — "
                  f"H1 not clearly supported. Linear angle performs as well as sin(θ).")
    a("")

    # ── H2 test: per-condition comparison ─────────────────────────────────────
    a("## 5. H2 Test — Static vs Dynamic Model Comparison by Condition")
    a("")
    a("Models run separately on STATIC rows and DYNAMIC rows. "
      "The key comparison within DYNAMIC is M2 (torque only) vs M3/M4 (torque + ω).")
    a("")
    if condition_comparison_df is not None and not condition_comparison_df.empty:
        display_cols = [c for c in
                        ["condition", "model_id", "label", "n_obs",
                         "r_squared", "adj_r_squared", "rmse", "cv_rmse_mean"]
                        if c in condition_comparison_df.columns]
        a(condition_comparison_df[display_cols].round(4).to_markdown(index=False))
    else:
        a("*No per-condition data available.*")
    a("")

    # ── Best model interpretation ─────────────────────────────────────────────
    best = _best_model(comparison_df)
    a("## 6. Overall Model Comparison")
    a("")
    if best:
        a(f"The model with the lowest cross-validated RMSE was **{best['model_id']}** "
          f"({best.get('label', '')}).")
        a(f"It explained approximately **{best.get('r_squared', 'N/A'):.1%}** of the "
          f"variance in {DEPENDENT_VAR} (R² = {best.get('r_squared', 'N/A')}).")
        a("")
        a("> **Caution:** These results are based on a single-participant, "
          "repeated-measures dataset. Regression associations described here "
          "should not be interpreted as population-level causal relationships. "
          "Replication across multiple participants would be required before "
          "generalizing these findings.")
    else:
        a("*Insufficient data to determine best model.*")
    a("")

    # ── Segmented regression ──────────────────────────────────────────────────
    a("## 7. H3 Test — Threshold Analysis (Segmented Regression)")
    a("")
    if seg_results.get("success"):
        theta = seg_results["theta_star_deg"]
        s_bel = seg_results["slope_below"]
        s_abv = seg_results["slope_above"]
        a(f"A piecewise linear model identified a potential activation threshold at:")
        a(f"")
        a(f"**θ\\* = {theta:.1f}°** (neck flexion angle)")
        a("")
        a(f"| Segment | Slope (%%MVC per °) |")
        a(f"|---------|-------------------|")
        a(f"| Below θ\\* | {s_bel:.4f} |")
        a(f"| Above θ\\* | {s_abv:.4f} |")
        a("")
        if abs(s_abv) > abs(s_bel):
            a("The slope above the threshold is steeper than below, which is "
              "consistent with the hypothesis of disproportionate neuromuscular "
              "activation beyond a critical flexion angle.")
        else:
            a("The slope above the threshold does not appear steeper than below. "
              "Evidence for a disproportionate increase in activation at this angle "
              "is not clear from the current data.")
        a("")
        a("> **Note:** Change-point detection is sensitive to sample size and "
          "data distribution. This threshold estimate should be treated as "
          "exploratory and hypothesis-generating.")
    else:
        a(f"Segmented regression could not be completed: "
          f"{seg_results.get('reason', 'unknown reason')}.")
    a("")

    # ── Limitations ───────────────────────────────────────────────────────────
    a("## 8. Limitations")
    a("")
    a("- Data collected from a single participant (the researcher); "
      "results cannot be generalized to other individuals.")
    a("- Cervical torque is estimated from a static biomechanical model "
      "(T = F × d × sin θ) and does not account for individual anatomy or "
      "muscle co-contraction.")
    a("- Angular acceleration was collected by the firmware but was not "
      "used as a regression predictor in this pipeline, consistent with "
      "the pre-specified analysis plan.")
    a("- EMG normalization depends on the quality of the MVC calibration trial.")
    a("- Regression models assume linear relationships; nonlinear effects "
      "beyond the segmented regression are not fully captured.")
    a("")

    report_path = reports_dir / "analysis_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("  Saved: analysis_summary.md")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _best_model(comparison_df: pd.DataFrame) -> Optional[dict]:
    """Return the row dict of the model with the lowest cv_rmse_mean."""
    if comparison_df is None or comparison_df.empty:
        return None
    if "cv_rmse_mean" not in comparison_df.columns:
        return None
    valid = comparison_df.dropna(subset=["cv_rmse_mean"])
    if valid.empty:
        return None
    return valid.loc[valid["cv_rmse_mean"].idxmin()].to_dict()


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy scalars and NaN."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_html_report(
    df:                      pd.DataFrame,
    comparison_df:           pd.DataFrame,
    coeff_df:                pd.DataFrame,
    seg_results:             dict,
    quality_report:          dict,
    file_stats:              List[dict],
    figures_dir:             Path,
    reports_dir:             Path,
    condition_comparison_df: pd.DataFrame = None,
    per_condition:           dict = None,
    seg_results_dynamic:     dict = None,
) -> None:
    """
    Generate a self-contained HTML report of all findings and conclusions.

    All figures are embedded as base64 so the file has no external dependencies.
    No new libraries required beyond the existing requirements.

    Parameters
    ----------
    df             : processed DataFrame
    comparison_df  : model comparison table
    coeff_df       : regression coefficients table
    seg_results    : segmented regression results dict
    quality_report : data quality report dict
    file_stats     : per-file loading statistics
    figures_dir    : directory where PNG figures were saved
    reports_dir    : directory where the HTML file will be saved
    """
    import base64

    reports_dir.mkdir(parents=True, exist_ok=True)
    best = _best_model(comparison_df)

    # ── Helper: embed a PNG as a base64 img tag ───────────────────────────────
    def _img(filename: str, caption: str = "") -> str:
        path = figures_dir / filename
        if not path.exists():
            return f'<p class="missing">Figure not available: {filename}</p>'
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return (
            f'<figure>'
            f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )

    # ── Helper: DataFrame → HTML table ────────────────────────────────────────
    def _table(data: pd.DataFrame, highlight_col: str = "", highlight_fn=None) -> str:
        if data is None or data.empty:
            return "<p><em>No data available.</em></p>"
        rows_html = ""
        for _, row in data.iterrows():
            highlight = (
                highlight_fn is not None and
                highlight_col and
                highlight_col in row and
                highlight_fn(row)
            )
            tr_class = ' class="best-row"' if highlight else ""
            cells = "".join(
                f"<td>{_fmt(v)}</td>" for v in row.values
            )
            rows_html += f"<tr{tr_class}>{cells}</tr>\n"
        headers = "".join(f"<th>{c}</th>" for c in data.columns)
        return (
            f'<div class="table-wrap"><table>'
            f'<thead><tr>{headers}</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>'
        )

    def _fmt(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # ── Derive conclusion strings ─────────────────────────────────────────────

    # 1. Data quality conclusion
    n_valid   = quality_report.get("n_valid_rows", 0)
    n_input   = quality_report.get("n_input_rows", 0)
    n_sessions = quality_report.get("sessions", 1)
    pct_kept  = (n_valid / n_input * 100) if n_input > 0 else 0
    dq_conclusion = (
        f"Data loading was successful. {n_valid:,} of {n_input:,} rows "
        f"({pct_kept:.1f}%) passed all quality filters across {n_sessions} session(s). "
        f"{'The high retention rate suggests clean firmware output and a valid MVC calibration.' if pct_kept > 90 else 'A notable number of rows were discarded; verify MVC calibration and sensor status during recording.'}"
    )

    # 2. Descriptive statistics conclusion
    desc_conclusion = "No data available for descriptive statistics."
    if "angle_deg" in df.columns and "emg_percent_mvc" in df.columns:
        angle_range = df["angle_deg"].max() - df["angle_deg"].min()
        mvc_mean    = df["emg_percent_mvc"].mean()
        mvc_std     = df["emg_percent_mvc"].std()
        desc_conclusion = (
            f"The dataset covers a neck flexion range of {angle_range:.1f}°, "
            f"which {'provides adequate variation for regression modeling' if angle_range > 10 else 'may be too narrow for reliable regression — consider a wider range of motion'}. "
            f"Mean EMG activation was {mvc_mean:.1f}% MVC (SD = {mvc_std:.1f}%), "
            f"indicating {'moderate' if mvc_mean < 50 else 'high'} muscle recruitment across the recorded trials."
        )

    # 3. Model comparison conclusion
    if best:
        best_id    = best.get("model_id", "")
        best_r2    = best.get("r_squared", float("nan"))
        best_cvrmse = best.get("cv_rmse_mean", float("nan"))
        is_dynamic = "velocity" in best_id.lower()
        mc_conclusion = (
            f"The best-performing model was <strong>{best.get('label', best_id)}</strong> "
            f"(R² = {best_r2:.3f}, CV-RMSE = {best_cvrmse:.3f}). "
            + (
                "The inclusion of angular velocity alongside torque improved predictive accuracy, "
                "providing preliminary support for Hypothesis H2: that dynamic kinematic parameters "
                "explain additional variance in cervical muscle activation beyond static angle-based models."
                if is_dynamic else
                "A static torque or angle model performed best, suggesting that — for this dataset — "
                "the addition of angular velocity did not substantially improve prediction. "
                "This may reflect limitations in data variability or task design rather than a true absence of dynamic effects."
            )
        )
    else:
        mc_conclusion = "Insufficient data to determine the best model."

    # 3b. H1 conclusion — M1 (linear angle) vs M2 (torque = sin θ)
    h1_conclusion = "M1 vs M2 comparison not available (insufficient model data)."
    if not comparison_df.empty:
        m1_row = comparison_df[comparison_df["model_id"] == "M1_angle"]
        m2_row = comparison_df[comparison_df["model_id"] == "M2_torque"]
        if not m1_row.empty and not m2_row.empty:
            m1_r2 = float(m1_row["r_squared"].values[0] or 0)
            m2_r2 = float(m2_row["r_squared"].values[0] or 0)
            delta  = m2_r2 - m1_r2
            if delta > 0:
                h1_conclusion = (
                    f"M2 (torque ∝ sin θ, R²={m2_r2:.3f}) outperforms M1 "
                    f"(linear angle, R²={m1_r2:.3f}) by ΔR²={delta:.3f}. "
                    f"This supports <strong>H1</strong>: EMG activation is better described by a "
                    f"nonlinear (sinusoidal) function of flexion angle than by a linear one."
                )
            else:
                h1_conclusion = (
                    f"M1 (linear angle, R²={m1_r2:.3f}) performs as well as M2 "
                    f"(torque ∝ sin θ, R²={m2_r2:.3f}, ΔR²={delta:.3f}). "
                    f"H1 is not clearly supported in this dataset; the relationship "
                    f"appears approximately linear over the recorded angle range."
                )

    # 3c. Enhance H2 conclusion with ω p-value from DYNAMIC coeff table
    cond_conclusion = ""
    if per_condition and "DYNAMIC" in per_condition:
        dyn_coeff = per_condition["DYNAMIC"].get("coeff_df", pd.DataFrame())
        if not dyn_coeff.empty:
            m3_omega = dyn_coeff[
                (dyn_coeff["model_id"] == "M3_torque_abs_velocity") &
                (dyn_coeff["term"] == "abs_angular_velocity_deg_s")
            ]
            if not m3_omega.empty:
                omega_p = m3_omega["p_value"].values[0]
                if omega_p is not None:
                    sig_str = (
                        f"significant (p = {omega_p:.4f} < 0.05)"
                        if float(omega_p) < 0.05
                        else f"not significant (p = {omega_p:.4f} ≥ 0.05)"
                    )
                    cond_conclusion += (
                        f" The |ω| coefficient in M3 is <strong>{sig_str}</strong> "
                        f"within the DYNAMIC condition."
                    )

    # 4. Regression coefficients conclusion
    coeff_conclusion = "Coefficient interpretation requires reviewing the table above."
    if coeff_df is not None and not coeff_df.empty:
        m1 = coeff_df[coeff_df["model_id"] == "M1_angle"]
        if not m1.empty and "angle_deg" in m1["term"].values:
            angle_coef = m1.loc[m1["term"] == "angle_deg", "coef"]
            if not angle_coef.empty and angle_coef.values[0] is not None:
                direction = "positive" if float(angle_coef.values[0]) > 0 else "negative"
                coeff_conclusion = (
                    f"The coefficient for <em>angle_deg</em> in the angle-only model (M1) is "
                    f"{direction}, consistent with the expectation that greater neck flexion "
                    f"is associated with {'higher' if direction == 'positive' else 'lower'} muscle activation. "
                    f"Standardized coefficients (model ID suffix _std) allow direct comparison of "
                    f"predictor importance across variables with different units."
                )

    # 5. Segmented regression conclusion (all data + DYNAMIC subset)
    dyn_seg_note = ""
    if seg_results_dynamic and seg_results_dynamic.get("success"):
        dt = seg_results_dynamic["theta_star_deg"]
        ds_b = seg_results_dynamic["slope_below"]
        ds_a = seg_results_dynamic["slope_above"]
        dyn_seg_note = (
            f" In the DYNAMIC-only subset, θ* = {dt:.1f}° "
            f"(slope below: {ds_b:.4f}, slope above: {ds_a:.4f})."
        )

    if seg_results.get("success"):
        theta   = seg_results["theta_star_deg"]
        s_below = seg_results["slope_below"]
        s_above = seg_results["slope_above"]
        steeper = abs(s_above) > abs(s_below)
        seg_conclusion = (
            f"A potential neuromuscular compensation threshold was identified at "
            f"<strong>θ* = {theta:.1f}°</strong>. "
            f"The slope of EMG activation vs. angle {'increases' if steeper else 'does not clearly increase'} "
            f"above this threshold (slope below: {s_below:.4f}; slope above: {s_above:.4f}). "
            + (
                "This is consistent with Hypothesis H3: that a flexion angle exists beyond which "
                "muscle activation rises disproportionately, potentially marking a critical "
                "stabilization demand threshold in the cervical spine."
                if steeper else
                "This does not strongly support Hypothesis H3. The current data do not clearly "
                "show a disproportionate activation increase above this angle. "
                "Additional trials or a wider angle range may be needed."
            ) +
            " <em>Note: this is an exploratory finding and should be interpreted cautiously "
            "given the single-participant design.</em>"
            + dyn_seg_note
        )
    else:
        seg_conclusion = (
            f"Segmented regression could not be completed: "
            f"{seg_results.get('reason', 'insufficient data')}. "
            "Consider collecting data across a wider range of neck flexion angles."
        )

    # 6. Overall conclusions
    overall_conclusion = (
        "This analysis examined whether dynamic kinematic parameters improve prediction "
        "of cervical muscle activation beyond static posture-based models. "
    )
    if best:
        if "velocity" in best.get("model_id", "").lower():
            overall_conclusion += (
                "Results provide preliminary support for the dynamic modeling hypothesis: "
                "a model incorporating angular velocity alongside cervical torque outperformed "
                "the static angle-only baseline. "
            )
        else:
            overall_conclusion += (
                "The static model performed competitively, and the benefit of adding angular "
                "velocity was not clearly demonstrated in this dataset. "
            )
    if seg_results.get("success"):
        overall_conclusion += (
            f"A change-point analysis identified a candidate flexion threshold at "
            f"{seg_results['theta_star_deg']:.1f}°, warranting further investigation. "
        )
    overall_conclusion += (
        "All findings are based on a single-participant repeated-measures design and "
        "should be treated as hypothesis-generating rather than conclusive."
    )

    # ── Build descriptive statistics table ───────────────────────────────────
    key_cols = [c for c in
                ["angle_deg", "angular_velocity_deg_s", "cervical_torque_nm", "emg_percent_mvc"]
                if c in df.columns]
    desc_html = "<p><em>No numeric data available.</em></p>"
    if key_cols:
        desc_html = _table(df[key_cols].describe().round(3).reset_index().rename(columns={"index": "stat"}))

    # ── Build per-condition comparison table (Static vs Dynamic) ─────────────
    cond_comp_html = "<p><em>No per-condition data available. Ensure CSV files contain STATIC and DYNAMIC status rows.</em></p>"
    cond_conclusion = ""
    if condition_comparison_df is not None and not condition_comparison_df.empty:
        display_cols = [c for c in
                        ["condition", "model_id", "label", "n_obs",
                         "r_squared", "adj_r_squared", "rmse", "aic", "cv_rmse_mean"]
                        if c in condition_comparison_df.columns]
        cond_comp_html = _table(condition_comparison_df[display_cols].round(4))

        # Build H2 conclusion: compare M2 vs M3 within DYNAMIC
        dyn_rows = condition_comparison_df[condition_comparison_df["condition"] == "DYNAMIC"]
        sta_rows = condition_comparison_df[condition_comparison_df["condition"] == "STATIC"]
        if not dyn_rows.empty and "cv_rmse_mean" in dyn_rows.columns:
            dyn_static = dyn_rows[dyn_rows["model_id"] == "M2_torque"]
            dyn_dynamic = dyn_rows[dyn_rows["model_id"].isin(["M3_torque_abs_velocity"])]
            if not dyn_static.empty and not dyn_dynamic.empty:
                best_static_rmse  = dyn_static["cv_rmse_mean"].min()
                best_dynamic_rmse = dyn_dynamic["cv_rmse_mean"].min()
                best_dynamic_id   = dyn_dynamic.loc[dyn_dynamic["cv_rmse_mean"].idxmin(), "model_id"]
                improvement = best_static_rmse - best_dynamic_rmse
                pct_improve = (improvement / best_static_rmse * 100) if best_static_rmse > 0 else 0
                if improvement > 0:
                    cond_conclusion = (
                        f"Within the <strong>DYNAMIC</strong> condition, adding angular velocity "
                        f"({best_dynamic_id}) reduced CV-RMSE by {improvement:.3f} "
                        f"({pct_improve:.1f}%) compared to torque-only (M2). "
                        f"This supports <strong>Hypothesis H2</strong>: dynamic kinematic parameters "
                        f"improve prediction of cervical muscle activation beyond static torque alone."
                    )
                else:
                    cond_conclusion = (
                        f"Within the <strong>DYNAMIC</strong> condition, adding angular velocity "
                        f"did not reduce CV-RMSE compared to torque-only (M2). "
                        f"H2 is not supported by the current data. Consider collecting more "
                        f"dynamic trials with a wider range of angular velocities."
                    )

    # ── Build comparison table with best row highlighted ──────────────────────
    comp_html = "<p><em>No model results available.</em></p>"
    if comparison_df is not None and not comparison_df.empty:
        display_cols = [c for c in
                        ["model_id", "label", "n_obs", "r_squared",
                         "adj_r_squared", "rmse", "aic", "cv_rmse_mean"]
                        if c in comparison_df.columns]
        best_id_val = best.get("model_id") if best else None
        comp_html = _table(
            comparison_df[display_cols].round(4),
            highlight_col="model_id",
            highlight_fn=lambda row: row.get("model_id") == best_id_val,
        )

    # ── Coefficient table (unstandardized only, cleaner for HTML) ─────────────
    coeff_html = "<p><em>No coefficient data available.</em></p>"
    if coeff_df is not None and not coeff_df.empty:
        unstd = coeff_df[~coeff_df["model_id"].str.endswith("_std")]
        if not unstd.empty:
            coeff_html = _table(unstd.round(4))

    # ── Segmented regression summary table ────────────────────────────────────
    seg_keys = ["theta_star_deg", "slope_below", "slope_above",
                "intercept", "n_below", "n_above", "n_total"]
    seg_html = "<p><em>Segmented regression not available.</em></p>"
    if seg_results.get("success"):
        seg_row = {k: seg_results[k] for k in seg_keys if k in seg_results}
        seg_html = _table(pd.DataFrame([seg_row]))

    # ── CSS ───────────────────────────────────────────────────────────────────
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px;
           color: #222; background: #f7f8fa; }
    .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
    h1 { font-size: 1.8em; color: #1a3a5c; border-bottom: 3px solid #2166ac;
         padding-bottom: 10px; margin-bottom: 6px; }
    h2 { font-size: 1.25em; color: #2166ac; margin: 32px 0 8px; border-left: 4px solid #2166ac;
         padding-left: 10px; }
    h3 { font-size: 1.05em; color: #444; margin: 18px 0 6px; }
    .meta { color: #666; font-size: 0.88em; margin-bottom: 28px; }
    .section { background: #fff; border-radius: 8px; padding: 20px 24px;
               margin-bottom: 22px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
    .conclusion { background: #eef4fb; border-left: 4px solid #2166ac;
                  border-radius: 0 6px 6px 0; padding: 12px 16px;
                  margin-top: 14px; font-size: 0.95em; line-height: 1.6; }
    .conclusion.warning { background: #fff8e1; border-left-color: #f59e0b; }
    .conclusion.success { background: #e8f5e9; border-left-color: #1a9641; }
    .table-wrap { overflow-x: auto; margin-top: 10px; }
    table { border-collapse: collapse; width: 100%; font-size: 0.88em; }
    th { background: #2166ac; color: #fff; padding: 8px 10px; text-align: left;
         white-space: nowrap; }
    td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; }
    tr:hover td { background: #f0f4ff; }
    tr.best-row td { background: #d4edda; font-weight: 600; }
    figure { margin: 16px 0; text-align: center; }
    figure img { max-width: 100%; border-radius: 6px;
                 box-shadow: 0 1px 6px rgba(0,0,0,0.12); }
    figcaption { font-size: 0.85em; color: #666; margin-top: 6px; }
    .fig-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .missing { color: #999; font-style: italic; }
    .badge { display: inline-block; background: #2166ac; color: #fff;
             border-radius: 4px; padding: 2px 8px; font-size: 0.82em; }
    .badge.green { background: #1a9641; }
    .limitations li { margin: 6px 0 6px 18px; line-height: 1.6; }
    footer { text-align: center; color: #aaa; font-size: 0.82em;
             margin-top: 40px; padding-top: 16px;
             border-top: 1px solid #e5e7eb; }
    """

    # ── HTML ──────────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    files_list = "".join(
        f"<li><code>{s['file']}</code> — {s.get('parsed_rows', 0):,} rows</li>"
        for s in file_stats
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cervical Neuromuscular Stabilization — Analysis Report</title>
<style>{css}</style>
</head>
<body>
<div class="container">

  <h1>Cervical Neuromuscular Stabilization<br>
      <small style="font-size:0.6em;color:#555;">Analysis Report</small></h1>
  <p class="meta">
    Generated: {now_str} &nbsp;|&nbsp;
    Greater Kansas City Science Fair &nbsp;|&nbsp;
    Vikram Vallurupalli
  </p>

  <!-- ── 1. Data Quality ──────────────────────────────────────────────── -->
  <div class="section">
    <h2>1. Data Quality Summary</h2>
    <ul style="margin: 8px 0 8px 20px; line-height:1.8">
      <li>Files processed: <strong>{len(file_stats)}</strong>
        <ul>{files_list}</ul>
      </li>
      <li>Total rows loaded: <strong>{n_input:,}</strong></li>
      <li>Valid rows after cleaning: <strong>{n_valid:,}</strong>
          ({pct_kept:.1f}% retained)</li>
      <li>Sessions: <strong>{n_sessions}</strong></li>
    </ul>
    <div class="conclusion {'success' if pct_kept > 90 else 'warning'}">
      <strong>Conclusion:</strong> {dq_conclusion}
    </div>
  </div>

  <!-- ── 2. Descriptive Statistics ────────────────────────────────────── -->
  <div class="section">
    <h2>2. Descriptive Statistics</h2>
    {desc_html}
    <div class="conclusion">
      <strong>Conclusion:</strong> {desc_conclusion}
    </div>
  </div>

  <!-- ── 3. Model Comparison ──────────────────────────────────────────── -->
  <div class="section">
    <h2>3. Regression Model Comparison</h2>
    <p style="font-size:0.85em;color:#666;margin-bottom:8px;">
      <span class="badge green">Best model</span> row is highlighted in green.
      CV-RMSE is the primary ranking metric (lower is better).
    </p>
    {comp_html}
    <div class="conclusion {'success' if best and 'velocity' in best.get('model_id','').lower() else ''}">
      <strong>Conclusion:</strong> {mc_conclusion}
    </div>
  </div>

  <!-- ── 3b. H1 Test: Nonlinearity ────────────────────────────────────── -->
  <div class="section">
    <h2>3b. H1 Test — Nonlinearity: sin(θ) vs Linear Angle</h2>
    <p style="font-size:0.85em;color:#666;margin-bottom:8px;">
      Cervical torque = F × d × sin(θ). Model M2 (torque) therefore fits
      EMG ~ sin(θ), a nonlinear function of angle. Comparing M1 (linear angle)
      to M2 directly tests H1.
    </p>
    <div class="conclusion {'success' if 'supports H1' in h1_conclusion else 'warning'}">
      <strong>H1 Conclusion:</strong> {h1_conclusion}
    </div>
  </div>

  <!-- ── 4. Static vs Dynamic Condition Comparison ───────────────────── -->
  <div class="section">
    <h2>4. Model Comparison by Condition (Static vs Dynamic) — H2 Test</h2>
    <p style="font-size:0.85em;color:#666;margin-bottom:8px;">
      Models run separately on STATIC rows and DYNAMIC rows.
      The key test: does adding angular velocity (M3/M4) improve on torque-only (M2)
      within the DYNAMIC condition?
    </p>
    {cond_comp_html}
    <div class="conclusion {'success' if cond_conclusion and 'supports' in cond_conclusion else 'warning'}">
      <strong>H2 Conclusion:</strong>
      {cond_conclusion if cond_conclusion else "Insufficient data in one or both conditions to evaluate H2."}
    </div>
  </div>

  <!-- ── 5. Regression Coefficients ──────────────────────────────────── -->
  <div class="section">
    <h2>5. Regression Coefficients</h2>
    <p style="font-size:0.85em;color:#666;margin-bottom:8px;">
      Unstandardized coefficients shown. p-values &lt; 0.05 indicate statistical
      significance at the 5% level. Standardized (beta) coefficients are available
      in <code>tables/regression_coefficients.csv</code>.
    </p>
    {coeff_html}
    <div class="conclusion">
      <strong>Conclusion:</strong> {coeff_conclusion}
    </div>
  </div>

  <!-- ── 6. Segmented Regression ──────────────────────────────────────── -->
  <div class="section">
    <h2>6. Segmented Regression — Flexion Threshold (θ*)</h2>
    {seg_html}
    {_img("segmented_threshold_plot.png",
          "Piecewise linear fit of EMG %MVC vs neck flexion angle with detected threshold θ*")}
    <div class="conclusion {'success' if seg_results.get('success') and abs(seg_results.get('slope_above',0)) > abs(seg_results.get('slope_below',0)) else 'warning'}">
      <strong>Conclusion:</strong> {seg_conclusion}
    </div>
  </div>

  <!-- ── 7. Time-Series Figures ────────────────────────────────────────── -->
  <div class="section">
    <h2>7. Time-Series Signals</h2>
    <div class="fig-grid">
      {_img("angle_vs_time.png",       "Neck flexion angle (°) over time")}
      {_img("velocity_vs_time.png",    "Angular velocity (°/s) over time")}
      {_img("torque_vs_time.png",      "Estimated cervical torque (Nm) over time")}
      {_img("emg_percent_mvc_vs_time.png", "EMG %MVC over time")}
    </div>
    <div class="conclusion">
      <strong>Conclusion:</strong>
      The time-series plots allow visual inspection of signal quality, motion patterns,
      and the correspondence between kinematic events and EMG responses. Peaks in angular
      velocity should visibly precede or coincide with peaks in EMG activation if dynamic
      stabilization demands are driving muscle recruitment.
    </div>
  </div>

  <!-- ── 8. Scatter Plots ──────────────────────────────────────────────── -->
  <div class="section">
    <h2>8. Predictor–Response Scatter Plots</h2>
    <div class="fig-grid">
      {_img("emg_vs_angle_scatter.png",    "EMG %MVC vs neck flexion angle with linear fit")}
      {_img("emg_vs_torque_scatter.png",   "EMG %MVC vs cervical torque with linear fit")}
      {_img("emg_vs_velocity_scatter.png", "EMG %MVC vs angular velocity with linear fit")}
    </div>
    <div class="conclusion">
      <strong>Conclusion:</strong>
      Scatter plots reveal the direction and linearity of each predictor–response
      relationship. A tighter cluster around the regression line indicates a stronger
      linear association. Non-random residual patterns (e.g., curved spread) would
      suggest nonlinear effects not captured by the linear models.
    </div>
  </div>

  <!-- ── 9. Predicted vs Actual ────────────────────────────────────────── -->
  <div class="section">
    <h2>9. Predicted vs Actual EMG %MVC</h2>
    <div class="fig-grid">
      {_img("predicted_vs_actual_static.png",
            "Model 2 (static torque): predicted vs actual %MVC")}
      {_img("predicted_vs_actual_dynamic.png",
            "Model 3/4 (dynamic): predicted vs actual %MVC")}
    </div>
    <div class="conclusion">
      <strong>Conclusion:</strong>
      Points closer to the dashed diagonal (y = x) indicate better model fit.
      Systematic deviation above or below the line reveals consistent over- or
      under-prediction. Comparing the static and dynamic panels directly illustrates
      whether adding angular velocity reduces prediction error in practice.
    </div>
  </div>

  <!-- ── 10. Condition Overlay ──────────────────────────────────────────── -->
  <div class="section">
    <h2>10. Experimental Condition Overlay</h2>
    {_img("optional_condition_overlay.png",
          "EMG %MVC over time with experimental condition shading")}
    <div class="conclusion">
      <strong>Conclusion:</strong>
      If condition labels (baseline / static / dynamic) were applied, this plot
      shows how muscle activation differs across experimental blocks. A visible
      increase in %MVC during dynamic compared to static conditions would provide
      qualitative support for the dynamic loading hypothesis.
      If all rows show as 'unlabeled', place a <code>conditions.csv</code> file
      in <code>data_raw/</code> and re-run the pipeline.
    </div>
  </div>

  <!-- ── 11. Overall Conclusions ──────────────────────────────────────── -->
  <div class="section">
    <h2>11. Overall Conclusions</h2>
    <div class="conclusion success">
      <strong>Summary:</strong> {overall_conclusion}
    </div>
    <h3 style="margin-top:18px;">Limitations</h3>
    <ul class="limitations">
      <li>Data collected from a single participant (the researcher);
          results cannot be generalized to other individuals without replication.</li>
      <li>Cervical torque is estimated from a static biomechanical model
          (T = 50 N × 0.08 m × sin θ) and does not account for individual
          anatomy or muscle co-contraction.</li>
      <li>Angular acceleration is captured by the firmware but was excluded
          from regression predictors per the pre-specified analysis plan.</li>
      <li>EMG normalization depends on the quality of the MVC calibration trial.</li>
      <li>All regression models assume linear relationships; nonlinear effects
          beyond the segmented regression are not fully characterized.</li>
    </ul>
  </div>

  <footer>
    Cervical Neuromuscular Stabilization &mdash; Greater Kansas City Science Fair
    &nbsp;|&nbsp; Vikram Vallurupalli &nbsp;|&nbsp; Generated {now_str}
  </footer>

</div>
</body>
</html>"""

    out_path = reports_dir / "analysis_report.html"
    out_path.write_text(html, encoding="utf-8")
    log.info("  Saved: analysis_report.html")
