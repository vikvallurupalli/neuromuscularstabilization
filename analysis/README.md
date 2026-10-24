# Cervical Neuromuscular Stabilization — Analysis Pipeline

Python analysis pipeline for the Greater Kansas City Science Fair project:
**"Nonlinear Dynamic Modeling of Cervical Neuromuscular Stabilization
Using Inertial Kinematics and Surface Electromyography"**

---

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Expected Input Format

Place one or more raw CSV files exported from the **CervicalLogger** ESP32-S3
firmware into the `data_raw/` folder.

**CSV header (updated firmware with angular acceleration):**

```
timestamp_ms,angle_deg,angular_velocity_deg_s,angular_accel_deg_s2,cervical_torque_nm,emg_env_raw,emg_raw_raw,emg_rms,emg_percent_mvc,accel_angle_deg,status
```

**CSV header (older firmware without angular acceleration):**

```
timestamp_ms,angle_deg,angular_velocity_deg_s,cervical_torque_nm,emg_env_raw,emg_raw_raw,emg_rms,emg_percent_mvc,accel_angle_deg,status
```

The pipeline handles both formats automatically.

**Notes:**
- Lines starting with `#` are comment lines from the firmware and are ignored.
- Rows where `status` is not `STREAMING` are discarded.
- Rows where `emg_percent_mvc` is ≤ 0 (uncalibrated MVC) are discarded.
- Multiple files are treated as separate sessions and merged.

---

## Optional: Condition Labels

To label time ranges with experimental conditions (baseline / static / dynamic),
place a file named `conditions.csv` inside `data_raw/` with this format:

```
session_id,start_ms,end_ms,condition
session_01,0,30000,baseline
session_01,30000,90000,static
session_01,90000,150000,dynamic
```

`session_id` must match the filename (without `.csv`) of the corresponding data file.
If this file is absent, all rows are labeled `unlabeled` and the pipeline still runs.

---

## Running the Pipeline

From the `analysis/` directory:

```bash
python src/main.py --input_dir data_raw --output_dir .
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--input_dir` | `data_raw` | Directory containing raw CSV files |
| `--output_dir` | `.` | Root directory for all outputs |
| `--conditions_file` | `data_raw/conditions.csv` | Optional condition labels CSV |
| `--verbose` / `-v` | off | Enable debug logging |

---

## Output Structure

```
analysis/
  data_processed/
    cleaned_combined.csv       Raw data after cleaning
    processed_combined.csv     With all derived feature columns
  figures/
    angle_vs_time.png
    velocity_vs_time.png
    torque_vs_time.png
    emg_percent_mvc_vs_time.png
    emg_vs_angle_scatter.png
    emg_vs_torque_scatter.png
    emg_vs_velocity_scatter.png
    predicted_vs_actual_static.png
    predicted_vs_actual_dynamic.png
    segmented_threshold_plot.png
    optional_condition_overlay.png
  tables/
    summary_statistics.csv
    model_comparison.csv
    regression_coefficients.csv
    cross_validation_results.csv
    segmented_regression_results.csv
  reports/
    analysis_summary.md        Plain-language report
    analysis_summary.json      Machine-readable key metrics
    data_quality_report.json   Row-count and cleaning diagnostics
```

---

## Models Fitted

| ID | Formula | Purpose |
|---|---|---|
| M1 | `%MVC ~ angle_deg` | Static angle-only baseline |
| M2 | `%MVC ~ cervical_torque_nm` | Static torque proxy |
| M3 | `%MVC ~ torque + angular_velocity` | Dynamic model (signed velocity) |
| M4 | `%MVC ~ torque + \|angular_velocity\|` | Dynamic model (absolute velocity) |

Each model reports: R², adjusted R², RMSE, MAE, AIC, and 5-fold cross-validated RMSE.

Standardized (beta) coefficients are also computed for direct comparison across models.

---

## Segmented Regression

The pipeline searches for a breakpoint angle **θ\*** (between the 10th and 90th
percentile of observed angles) that minimises residual error in a piecewise
linear fit of `%MVC ~ angle_deg`. The estimated slopes below and above θ\* are
reported in `tables/segmented_regression_results.csv` and visualised in
`figures/segmented_threshold_plot.png`.

---

## Assumptions and Limitations

- Data are from a single participant (the researcher). Results are not
  generalizable without replication.
- `angular_accel_deg_s2` is captured in the firmware output but is **not
  used as a regression predictor** in this pipeline, consistent with the
  pre-specified analysis plan.
- Cervical torque is a static biomechanical estimate: T = 50 N × 0.08 m × sin(θ).
- A minimum of 30 valid rows is required to run the analysis.

---

## Project Structure

```
analysis/
  src/
    main.py         Pipeline entry point (run this)
    config.py       All tunable constants and model definitions
    io_utils.py     Robust CSV loading from firmware output
    cleaning.py     Data validation and cleaning
    features.py     Derived column engineering and condition labeling
    modeling.py     OLS regression, cross-validation, segmented regression
    plotting.py     All matplotlib figures
    reporting.py    CSV tables, Markdown report, JSON summaries
  data_raw/         Place raw CSV files here
  requirements.txt
  README.md
```
