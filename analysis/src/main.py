"""
main.py
Entry point for the cervical neuromuscular stabilization analysis pipeline.

Usage:
    python src/main.py --input_dir data_raw --output_dir .

All output directories are created automatically.
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when run as `python src/main.py`
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import cleaning
import features
import io_utils
import modeling
import plotting
import reporting
from config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


log = logging.getLogger(__name__)


# ── Directory scaffolding ─────────────────────────────────────────────────────

def _setup_directories(output_dir: Path) -> dict:
    dirs = {
        "data_processed": output_dir / "data_processed",
        "figures":        output_dir / "figures",
        "tables":         output_dir / "tables",
        "reports":        output_dir / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cervical Neuromuscular Stabilization Analysis Pipeline\n"
            "Processes raw CervicalLogger CSV data and produces regression "
            "analyses, figures, and reports."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f"Directory containing raw .csv files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--conditions_file",
        type=Path,
        default=None,
        help=(
            "Optional CSV file mapping time ranges to condition labels "
            "(baseline/static/dynamic). If omitted, conditions are unlabeled."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    log.info("=" * 60)
    log.info("Cervical Neuromuscular Stabilization — Analysis Pipeline")
    log.info("=" * 60)
    log.info("Input  : %s", args.input_dir.resolve())
    log.info("Output : %s", args.output_dir.resolve())

    # ── Validate input directory ──────────────────────────────────────────────
    if not args.input_dir.exists():
        log.error(
            "Input directory '%s' does not exist. "
            "Create it and place your raw CSV files there.",
            args.input_dir,
        )
        sys.exit(1)

    # ── Create output directories ─────────────────────────────────────────────
    dirs = _setup_directories(args.output_dir)

    # ── Step 1: Load raw data ─────────────────────────────────────────────────
    log.info("")
    log.info("── Step 1: Loading raw CSV files ────────────────────────────")
    try:
        raw_df, file_stats = io_utils.load_all_sessions(args.input_dir)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    # ── Step 2: Clean data ────────────────────────────────────────────────────
    log.info("")
    log.info("── Step 2: Cleaning data ────────────────────────────────────")
    cleaned_df, quality_report = cleaning.clean_dataset(raw_df)

    cleaned_path = dirs["data_processed"] / "cleaned_combined.csv"
    cleaned_df.to_csv(cleaned_path, index=False)
    log.info("Cleaned data saved to: %s", cleaned_path)

    if cleaned_df.empty:
        log.error("No valid rows remain after cleaning. Cannot continue.")
        reporting.write_quality_report(quality_report, file_stats, dirs["reports"])
        sys.exit(1)

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    log.info("")
    log.info("── Step 3: Engineering features ─────────────────────────────")
    processed_df = features.compute_derived_columns(cleaned_df)

    # Optional condition labeling
    conditions_path = (
        args.conditions_file
        if args.conditions_file
        else args.input_dir / "conditions.csv"
    )
    cond_df = features.load_conditions_file(conditions_path)
    processed_df = features.apply_conditions(processed_df, cond_df)

    processed_path = dirs["data_processed"] / "processed_combined.csv"
    processed_df.to_csv(processed_path, index=False)
    log.info("Processed data saved to: %s", processed_path)

    # ── Step 4: Statistical modeling ──────────────────────────────────────────
    log.info("")
    log.info("── Step 4: Running regression models ────────────────────────")
    ols_results, comparison_df, coeff_df, cv_df = modeling.run_all_models(processed_df)

    log.info("")
    log.info("── Step 4b: Models by condition (Static vs Dynamic) ─────────")
    per_condition, condition_comparison_df = modeling.run_models_by_condition(processed_df)
    if not condition_comparison_df.empty:
        for cond, res in per_condition.items():
            cdf = res["comparison_df"]
            best_row = cdf.loc[cdf["cv_rmse_mean"].idxmin()] if "cv_rmse_mean" in cdf.columns and not cdf.empty else None
            if best_row is not None:
                log.info(
                    "  %s best model: %s | R²=%.3f | CV-RMSE=%.3f",
                    cond, best_row.get("model_id", "?"),
                    best_row.get("r_squared", float("nan")),
                    best_row.get("cv_rmse_mean", float("nan")),
                )

    log.info("")
    log.info("── Step 4c: Segmented regression on STATIC-only data (H3) ───")
    seg_results_dynamic: dict = {"success": False, "reason": "No DYNAMIC rows found"}
    static_df_seg = processed_df[processed_df["status"] == "STATIC"] \
        if "status" in processed_df.columns else processed_df
    if len(static_df_seg) < 20:
        log.warning("  Fewer than 20 STATIC rows — falling back to all data.")
        static_df_seg = processed_df
    seg_results, _seg_search_df = modeling.run_segmented_regression(
        static_df_seg, target_col="emg_percent_mvc"
    )
    log.info(
        "  STATIC-only theta*=%.2f deg | R2=%.4f | success=%s",
        seg_results.get("theta_star_deg", float("nan")),
        seg_results.get("r2", float("nan")),
        seg_results.get("success", False),
    )

    # Get in-sample predictions for plotting
    model_predictions = modeling.get_model_predictions(ols_results, processed_df)

    # ── Step 5: Generate figures ──────────────────────────────────────────────
    log.info("")
    log.info("── Step 5: Generating figures ───────────────────────────────")
    plotting.generate_all_plots(
        static_df_seg, model_predictions, seg_results, dirs["figures"]
    )

    # ── Step 6: Save tables ───────────────────────────────────────────────────
    log.info("")
    log.info("── Step 6: Saving tables ────────────────────────────────────")
    reporting.save_tables(
        processed_df, comparison_df, coeff_df, cv_df,
        seg_results, dirs["tables"],
        condition_comparison_df=condition_comparison_df,
    )

    # ── Step 7: Write reports ─────────────────────────────────────────────────
    log.info("")
    log.info("── Step 7: Writing reports ──────────────────────────────────")
    reporting.write_quality_report(quality_report, file_stats, dirs["reports"])
    reporting.write_json_summary(
        processed_df, comparison_df, seg_results, quality_report, dirs["reports"]
    )
    reporting.write_markdown_report(
        processed_df, comparison_df, seg_results,
        quality_report, file_stats, dirs["reports"],
        condition_comparison_df=condition_comparison_df,
    )
    reporting.generate_html_report(
        processed_df, comparison_df, coeff_df,
        seg_results, quality_report, file_stats,
        dirs["figures"], dirs["reports"],
        condition_comparison_df=condition_comparison_df,
        per_condition=per_condition,
        seg_results_dynamic=seg_results_dynamic,
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("")
    log.info("Output locations:")
    log.info("  Cleaned CSV  : %s", dirs["data_processed"] / "cleaned_combined.csv")
    log.info("  Processed CSV: %s", dirs["data_processed"] / "processed_combined.csv")
    log.info("  Figures      : %s", dirs["figures"])
    log.info("  Tables       : %s", dirs["tables"])
    log.info("  Reports      : %s", dirs["reports"])
    log.info("  HTML Report  : %s", dirs["reports"] / "analysis_report.html")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
