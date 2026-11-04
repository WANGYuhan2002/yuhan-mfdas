# Run_4_assign_overlaps_with_kurtosis — Documentation

## Script Classification for This Release

**Main pipeline (must run in order 2a → 2b)**:
- **[Main, Script 2a]** `2a_Classification_byCCs_with_kurtosis.py`: Filters by `max_stalta > 10`, de-duplicates overlapping multi-template detections → `Events_classified_AllTemp.csv`.
- **[Main, Script 2b]** `2b_Est_Mag_with_GR_law_check.py`: Magnitude calibration via NOA ML regression; Gutenberg-Richter analysis → `Events_classified_Real_Mag.csv` (★ final catalog).

**Threshold selection (optional, run before 2a)**:
- **[Optional]** `1b_decide_kurtosis_by_collapsed_no.py`: Plots event count vs. `max_stalta` threshold (0–40, step 0.2) to justify the choice of threshold = 10.

**Exploratory / ignorable**:
- `1_check_kurtosis_CCs.py`: Reads Output_4, writes figures to Output_6 — path label mismatch; exploratory.
- `1a_kurtosis_statitics.py`: Hardcoded to old `matched_filter_v6` path; cannot run in this release.

**Plotting / paper scripts (run after 2b, require external data)**:
- `2b_999_plot_amp2mag_C-Amp.py`: Extended magnitude regression (5×2) paper figures.
- `3a_Show_Classification_byCCs.py`: Three-catalog comparison / trigger time-series plots (reads `IN_STALTA_cata/sta_lta_catalog.csv`, in the Zenodo archive).
- `3b_Show_withMag_bvalue.py`: G-R b-value time series.
- `4_FMD_moving_time_window.py`: Frequency-magnitude distribution (broken: `home_dir` hardcoded to legacy `matched_filter_v6` path).
- `5_Show_withMag_noise_Mc_bvalue_slip_momentum.py`: 4-panel per-template figure with noise level overlay (reads `IN_noise_meteo/noise_trends_f8-20Hz_p5.h5`, produced by the companion repository `WANGYuhan2002/yuhan_das_spectra` / `check_minmag_pkg/products/` in `matched_filter_release.zip`).
- `5_extract_Show_slip_momentum.py`: Slimmed derivative of `5_Show_withMag_...` that plots only the binned + cumulative slip panels across the 5 templates (other panels commented out; ignorable).
- `5_999_Show_slip_with_template_for_paper.py`: Paper-quality 5×2 template waterfall + cumulative slip figure (appends `external/GRL_DAS_MF_plots/functions/` as a fallback on `sys.path`, while `plot`, `load_das_h5_PE2024`, and `compute_correlation` resolve from this repository's `functions/` first; also requires raw DAS in `IN_DAS_data/continuous_scan_1h/`).

---

## I. Folder Overview and Position in the Overall Pipeline

This folder produces the **final earthquake catalog** for the entire MF pipeline. It reads trigger pkl files from Output_2, applies quality filters, de-duplicates overlapping detections, estimates magnitudes via NOA ML calibration, and computes Gutenberg-Richter statistics.

**Upstream sources**:
- Output_2 trigger pkl files (`level_20/` in the formal run).
- Output_3 summary CSV `Trig_MAX_amp_df.csv` (for optional `Quasi_Mag_Amp` branch).
- `Output_0_Template_library/Template_library.csv` (for `self_CC` and `amp_max_mean` template constants).
- `IN_NOA_cata/200km_distance/IMS_*.txt` (for magnitude calibration and comparison).

**Outputs** (in `Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_20/`):
- Script 2a: `Events_classified_AllTemp.csv` + `Events_classified_AllTemp_addRAWCCs.csv` + `config.json`.
- Script 2b: `MagRegression/Events_classified_Real_Mag.csv` (final catalog) + NOA match tables + GR figures.

**Pipeline position**:
```
Output_2 (triggers) + Output_3 (amplitudes) + Template_library.csv + NOA 200km
        │
        ▼
Script 2a → Events_classified_AllTemp.csv   (17,326 detections (17,325 within the analysis window ending 2024-06-22 16:15; 17,263 after merging overlapping detections across templates, Belong_or_not=1))
        │
        ▼
Script 2b → Events_classified_Real_Mag.csv  (adds Real_Mag; final catalog)
        │
        ▼
Run_6 (three-catalog comparison)
```

---

## II. Script-by-Script Description (Main Pipeline)

### Script `1b_decide_kurtosis_by_collapsed_no.py` (Optional)

- Sweeps `max_stalta` threshold from 0 to 40 in steps of 0.2; plots event count vs. threshold curve.
- Output: `stalta_threshold_analysis.csv` and figures in `Run_4_assign_overlaps_with_kurtosis/Output_1b_decide_kurtosis/level_20/`.
- Used to justify the choice of `max_stalta_threshold = 10` (chosen from the inflection point of the event-count curve).
- Can be run independently before Script 2a; does not enter the main data stream.

---

### Script 2a: `2a_Classification_byCCs_with_kurtosis.py` (Main — De-duplication and Classification)

- **Purpose**: Load all trigger pkl files from Output_2; apply `max_stalta > 10` filter; de-duplicate cross-template overlaps; assign `Belong_or_not`.
- **Inputs**:
  - Output_2 trigger pkl files (`level_20/` subdirectory).
  - `Template_library.csv` (for template time-offset correction via `Remove_Temp_Offest_forSeries`).
- **Processing logic**:
  1. `Read_Collapse_group_df(pkl_dir, Template_library_csv, ...)` → single DataFrame of all trigger events across all templates with all pkl fields.
  2. Apply `max_stalta > 10` mask (removes events with sharp-but-spiky STA/LTA peaks, primarily noise artifacts).
  3. `Collapse_with_MAX_CCs(df, time_window_s=2.5)` → within any 2.5-second window, keep only the detection with the highest `|CCs_s_trig|`; assign `Belong_or_not=1` to winners, `0` to losers; assign `Group_ID` and `Winner_Template`.
  4. Write `Events_classified_AllTemp.csv` (17,326 detections (17,325 within the analysis window ending 2024-06-22 16:15; 17,263 after merging overlapping detections across templates, i.e. `Belong_or_not=1`)); also re-read pkl files to add `max_RAWCCs` field → write `Events_classified_AllTemp_addRAWCCs.csv` (identical content, different name).
  5. Save `config.json` with parameters.
- **Key parameter**: `max_stalta_threshold = 10` (events with `max_stalta ≤ 10` receive `Belong_or_not=0`).

---

### Script 2b: `2b_Est_Mag_with_GR_law_check.py` (Main — Magnitude Calibration and G-R Law)

- **Purpose**: Estimate `Real_Mag` for all classified events via regression against NOA ML; compute G-R statistics.
- **Inputs**:
  - `Events_classified_AllTemp.csv` (Script 2a output).
  - `Output_3_Trig_PeakAmp/level_XX/Trig_MAX_amp_df.csv` (for `amp_log_mean`/`amp_max_mean`; optional `Quasi_Mag_Amp` branch).
  - `Template_library.csv`: columns `self_CC`, `amp_max_mean` (template constants).
  - `IN_NOA_cata/200km_distance/IMS_*.txt`: NOA catalog for regression.

**Magnitude formula (method='max_RAWCCs')**:

```python
# Step 1: template calibration factor (constant per template, from Template_library.csv)
fac_CC2amp = amp_max_mean_template / self_CC_template

# Step 2: quasi-magnitude per event
Quasi_Mag = log10(|max_RAWCCs_event × fac_CC2amp|)

# Step 3: regression against NOA ML
Real_Mag = slope × Quasi_Mag + intercept
```

**NOA matching** (function `_find_Template_EQs_in_IMS`):
- Spatial tolerance: ±0.05° latitude AND ±0.05° longitude from template hypocenter.
- Temporal tolerance: ±5 s between MF trigger and NOA origin time.
- Minimum match count per template: ≥3 matches required before fitting the regression.

**Outputs** (`MagRegression/` subdirectory):
- `Events_classified_Real_Mag.csv`: complete catalog with added columns `amp_log_mean, amp_max_mean, Quasi_Mag_Amp, fac_CC2amp, Quasi_Mag, Real_Mag`.
- `Events_with_NOA_Mag/NOA_match_<template>.csv`: per-template matched pairs used for regression.
- `GR_law_check/*.png`: magnitude-frequency distribution (FMD) check figures.
- `max_RAWCCs_vs_peakAmp/*.png`: scatter plots verifying correlation between `max_RAWCCs` and peak DAS amplitude.

---

## III. Key Formula Details

### `fac_CC2amp` — Template Calibration Factor

This converts raw cross-correlation amplitude to physical strain-rate amplitude:

```
fac_CC2amp[template] = amp_max_mean[template] / self_CC[template]
```

where:
- `amp_max_mean[template]`: mean per-channel peak amplitude when the template earthquake occurred (measured from DAS, 8.3–14 km channels, 8–20 Hz, from `Template_library.csv`).
- `self_CC[template]`: self cross-correlation amplitude (normalized; also from `Template_library.csv`).

Both values are **template-level constants** — they are computed once from the template event's own waveform and stored in `Template_library.csv`. They do **not** vary per detected event.

### `Quasi_Mag` vs. `Real_Mag`

| Quantity | Formula | Notes |
|---|---|---|
| `Quasi_Mag` | `log10(\|max_RAWCCs × fac_CC2amp\|)` | Dimensionless proxy in log-amplitude scale |
| `Real_Mag` | `slope × Quasi_Mag + intercept` | Calibrated against NOA ML via linear regression |
| `Quasi_Mag_Amp` | `amp_log_mean` from Output_3 | Alternative proxy using per-event DAS amplitude; stored but not used as final magnitude |

---

## IV. b-value Computation Parameters

Script 3b and the 5_ series use different parameters — **both are intentional**:

| Script | `min_events` | `step` (time bin) | `mc` method |
|---|---|---|---|
| `3b_*.py` | 80 | 0.2 h | `mc = minmag` (minimum magnitude in window) |
| `5_Show_*.py`, `5_extract_*.py` | 100 | 2 h | `mc = maxc` (maximum-curvature method) |

The two scripts serve different purposes: Script 3b provides fine temporal resolution (0.2h bins) for time-series analysis; the 5_ series provides coarser bins (2h) optimized for stable b-value estimates using the maximum-curvature Mc method.
