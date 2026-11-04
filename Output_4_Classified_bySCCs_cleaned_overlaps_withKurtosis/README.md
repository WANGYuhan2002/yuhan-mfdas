# Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis — Documentation

> **GitHub version:** the data files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> Release path: `yuhan_dasmf_v1.0/Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/`
> All contents produced by scripts in `Run_4_assign_overlaps_with_kurtosis/`.

---

## I. Directory Role

This directory contains the **final earthquake catalog** of the MF pipeline. It is produced by Run_4 scripts and is the terminal output of the main detection pipeline. It includes:
- De-duplicated event classification (Script 2a)
- Magnitude estimation and Gutenberg-Richter analysis (Script 2b)
- Quality-control figures (Scripts 3a, 3b, 5_Show, 5_extract, etc.)

Downstream consumer: Run_6 (three-catalog comparison) reads `Events_classified_Real_Mag.csv` from this directory.

---

## II. Directory Structure

```
Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/
└── level_20/
    ├── config.json                           # Script 2a parameters
    ├── Events_classified_AllTemp.csv         # ★ Script 2a output (17,326 detections; see Row count below)
    ├── Events_classified_AllTemp_addRAWCCs.csv  # Same content as above (identical)
    └── MagRegression/
        ├── Events_classified_Real_Mag.csv    # ★★ Script 2b output — FINAL CATALOG
        ├── Events_with_NOA_Mag/              # Per-template NOA ML matching tables
        │   ├── NOA_match_template_0423_80Hz.csv  (195 matches)
        │   ├── NOA_match_template_10_80Hz.csv    (121 matches)
        │   ├── NOA_match_template_0425_80Hz.csv  (38 matches)
        │   ├── NOA_match_template_8_80Hz.csv     (32 matches)
        │   └── NOA_match_template_20_80Hz.csv    (4 matches)
        ├── GR_law_check/                     # Gutenberg-Richter analysis PNGs
        └── max_RAWCCs_vs_peakAmp/            # Correlation check PNGs
```

`level_20` corresponds to the MAD×20 trigger threshold level used in Run_2; consistent with the `Threshold_level` parameter propagated through Run_2 → Run_3 → Run_4.

---

## III. Core CSV Files

### `Events_classified_AllTemp.csv` and `Events_classified_AllTemp_addRAWCCs.csv`

**Produced by**: Run_4 Script 2a.
**Content**: Identical (the `_addRAWCCs` version simply re-reads the pkl files to add the `max_RAWCCs` field — the core content is the same as the non-suffixed version).
**Row count**: 17,326 detections (17,325 within the analysis window ending 2024-06-22 16:15; 17,263 after merging overlapping detections across templates, i.e. `Belong_or_not == 1`).

**Columns** (17 columns):

| Column | Description |
|---|---|
| `CCs_s_trig` | SCCs value at trigger peak (≈ ±0.01) |
| `RAWCCs_s_trig` | Raw cross-correlation at trigger peak |
| `t_trig` | Trigger time (UTC datetime string) |
| `width_trig` | Trigger pulse width |
| `t_left / t_right` | Trigger window boundaries |
| `max_RAWCCs` | Max `\|RAWCCs\|` within trigger window |
| `t_arg_max` | Time of max RAWCCs |
| `max_stalta` | **STA/LTA ratio at trigger point** (quality criterion; events with `max_stalta ≤ 10` are excluded) |
| `Rt_trig` | Trigger time as string key `YYYYMMDDTHHMMSS` |
| `Rt_left / Rt_right` | Window boundaries as string keys |
| `CCs_s_trig_ofMAD` | `CCs_s_trig / MAD` (normalized) |
| `Template` | Template name (e.g. `template_10_80Hz`) |
| `Belong_or_not` | **1 = valid event** (passed max_stalta > 10 filter AND overlap de-duplication); 0 = rejected |
| `Group_ID` | Overlap group identifier (events that overlapped across templates) |
| `Winner_Template` | Template with highest `\|CCs_s_trig\|` in the overlap group |

**Filters applied by Script 2a**:
1. `max_stalta > 10` (removes impulsive noise / false positives).
2. `Collapse_with_MAX_CCs(2.5 s)` — within any 2.5-second window, keep only the detection with the highest `|CCs_s_trig|`; assign `Belong_or_not` to indicate winners.

---

### `Events_classified_Real_Mag.csv` ★★ — Final Earthquake Catalog

**Produced by**: Run_4 Script 2b.
**Content**: Extends `Events_classified_AllTemp.csv` with magnitude estimates and additional fields. Contains all rows, including `Belong_or_not==0`: 17,326 detections (17,325 within the analysis window ending 2024-06-22 16:15; 17,263 after merging overlapping detections across templates; apply the `Belong_or_not==1` filter to select these).

**Additional columns added by Script 2b**:

| Column | Description |
|---|---|
| `amp_log_mean` | Mean log-amplitude from Output_3 (optional Quasi_Mag_Amp branch) |
| `amp_max_mean` | Mean peak amplitude from Output_3 |
| `Quasi_Mag_Amp` | `amp_log_mean` as amplitude-based magnitude proxy (optional) |
| `fac_CC2amp` | Template calibration factor: `amp_max_mean_template / self_CC_template` (from `Template_library.csv`; template constant, NOT per-event) |
| `Quasi_Mag` | `log10(\|max_RAWCCs × fac_CC2amp\|)` — RAWCCs-based quasi-magnitude |
| `Real_Mag` | **★ Final calibrated magnitude**: `slope × Quasi_Mag + intercept` from linear regression against NOA ML |

**Magnitude calibration procedure** (Script 2b):
1. Match MF events to NOA IMS catalog (200km radius): tolerance ±0.05° lat/lon + 5 s time; ≥3 matches required per template.
2. Fit linear regression: `Real_Mag = slope × Quasi_Mag + intercept`.
3. Apply regression coefficients to all events.

---

### `Events_with_NOA_Mag/` — Per-Template NOA Matching Tables

One CSV per template, containing the matched MF event–NOA event pairs used for the magnitude regression:

| Template | Matches |
|---|---|
| template_0423_80Hz (K2) | 195 |
| template_10_80Hz (K1) | 121 |
| template_0425_80Hz (I2) | 38 |
| template_8_80Hz (K3) | 32 |
| template_20_80Hz (I1) | 4 |

---

## IV. Plotting Scripts and Their Outputs

Several plotting and analysis scripts in Run_4 produce additional figures in this directory:

| Script | Outputs | Notes |
|---|---|---|
| Script `2b_999_plot_amp2mag_C-Amp.py` | GR-law and magnitude regression PNGs | `999_plot_regression_CviaAmp_for_paper/` |
| Script `3a_Show_Classification_byCCs.py` | Time-series comparison with STA/LTA | Reads `IN_STALTA_cata/sta_lta_catalog.csv` (from the Zenodo archive) |
| Script `3b_Show_withMag_bvalue.py` | Gutenberg-Richter b-value time series | `min_events=80, step=0.2h, mc=minmag` |
| Script `5_Show_withMag_noise_Mc_bvalue_slip_momentum.py` | 4-panel noise level / Mc / b-value / slip / moment overlay | Reads `IN_noise_meteo/noise_trends_f8-20Hz_p5.h5` (in the Zenodo archive and in companion repo `WANGYuhan2002/yuhan_das_spectra` under `products/` / Zenodo `check_minmag_pkg/products/`) |
| Script `5_extract_Show_slip_momentum.py` | Combined 5-template binned + cumulative slip figure | Slimmed derivative of `5_Show_withMag_...` (`5_extract_slip_momentum/MF_slip_all.png`) |
| Script `5_999_Show_slip_with_template_for_paper.py` | Paper-quality 5×2 template waterfall + slip figure | Appends `external/GRL_DAS_MF_plots/functions/` as a fallback on `sys.path` (`plot`, `load_das_h5_PE2024`, `compute_correlation` resolve from this repository's `functions/` first); also reads raw DAS from `IN_DAS_data/continuous_scan_1h/` |
| Script `4_FMD_moving_time_window.py` | FMD analysis | Broken: `home_dir` hardcoded to legacy `matched_filter_v6` path |

**Note**: Scripts 5_ use different b-value parameters from Script 3b:
- Script 3b: `min_events=80, step=0.2h, mc=minmag`.
- Scripts 5_: `min_events=100, step=2h, mc=maxc` (maximum curvature method).
This difference is intentional — different temporal resolutions and Mc estimation methods.

---

## V. Physical Quantities Computed in Plotting Scripts (Not in CSV)

The Brune (1970) source model quantities are computed within plotting scripts for visualization but are **not stored in any CSV**:

- **Slip**: `Slip = (σ^(2/3) / μ) × 571.73 × 10^(0.5 × Mw)` (m)
  - σ: stress drop (typically 3 MPa for this region)
  - μ: shear modulus (~30 GPa)
- **Seismic moment**: `Moment = 10^(1.5 × (Mw + 6.03))` (N·m)
- **Direction**: inferred from `CCs_s_trig` sign — positive SCCs = same polarity as template (same fault mechanism); negative = opposite polarity.

These are computed per-event in figure-generating scripts only; if needed for analysis, re-run the corresponding plotting script.
