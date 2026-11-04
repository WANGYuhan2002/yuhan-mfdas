# Run_1_compute_SCCs_and_MAD — Documentation

## Script Classification for This Release

### [Main Line]

- `1_Compute_SCCs_10min_h5_Parallel_io_once.py`: Main version for SCC computation. Reads DAS continuous data only once per 10-minute window; all templates reuse that data segment, significantly reducing disk I/O. Works with 5 `template_*_80Hz` templates.
- `2_MAD_of_SCCs_calculation_Parallel.py`: Main version for MAD and statistics computation. Reads SCCs h5 files in parallel, computes 5 statistical quantities per fixed time window, writes pkl files.

### [Legacy (ignorable, kept in place)]

- `1_Compute_SCCs_10min_h5_Parallel.py`: Historical v3 version. Reads disk repeatedly for each `(template, window)` pair; paths point to `matched_filter_v3`; uses 13 old templates (`temp_0..temp_12`); outputs to `SCCs_h5_6h/`; inconsistent with this release's data. Cannot be run to reproduce existing results.
- `2_MAD_of_SCCs_calculation.py`: Historical v3 version. Paths point to `matched_filter_v3`; the computation body has been commented out entirely; when run, it only loads existing pkl + plots. One-off exploration script.

---

## I. Folder Overview and Position in the Overall Pipeline

This folder is **Step 1** of the template matching pipeline, responsible for two tasks:

1. **Computing Stacked Cross-Correlation Coefficients (SCCs)**: Using the template library from Output_0, slide a normalized cross-correlation along Kefalonia DAS continuous data, then average (stack) over the channel dimension (space) to obtain a time-varying correlation coefficient curve.
2. **Computing MAD (Median Absolute Deviation)**: Compute MAD over SCCs curves within fixed time windows as the noise baseline for subsequent trigger detection (trigger threshold = MAD × level).

**Upstream sources**:

- Template `.npy` files: `{PROJECT_ROOT}/Output_0_Template_library/Template_npy_files/` (produced by Run_0).
- Continuous DAS data: the 80 Hz archive `DATA_DEC_80Hz_16m_SORTED/` (80 Hz, 16 m channel spacing, external raw data, not produced by this project), read from the package-relative default `IN_DAS_data/continuous_scan_1h/` via `functions/readers.py`'s `Read_DAS_data` (place or symlink the archive, or the 1-hour sample from the Zenodo archive, there).

**Downstream destinations**:

- `{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/<template_name>/CCs_stack_*.h5`: Read by subsequent steps **Run_2** (triggering), **Run_3** (peak amplitude), etc. via `functions/readers.py`'s `Read_CCs_stack_h5`.
- `{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/.../MAD_Average_dict.pkl`: Read by **Run_2** as the trigger threshold scale (Run_2 reads `12:00:00/MAD_Average_dict.pkl`).

Overall pipeline: `Run_0 (template library) → [Run_1: this folder, SCCs + MAD] → Run_2 (triggering) → Run_3 (peak amplitude) → Run_5/6 (subsequent analysis)`.

---

## II. Script-by-Script Description

### Script 1: `1_Compute_SCCs_10min_h5_Parallel_io_once.py` [Main, SCC Computation]

- **Purpose**: Divide the entire observation period into 10-minute windows; each window reads DAS data **only once**, then computes stacked cross-correlation SCCs for all 5 templates; stores results per window per template as h5 files.
- **Inputs**:
  - Templates: 5 `.npy` files under `Output_0_Template_library/Template_npy_files/` (`template_0423_80Hz` / `template_0425_80Hz` / `template_10_80Hz` / `template_8_80Hz` / `template_20_80Hz`); all loaded into memory at once by `preload_all_templates` before starting.
  - Continuous data: 80 Hz DAS archive (`IN_DAS_data/continuous_scan_1h/`), read by `Read_DAS_data(t_start, t_end_for_read, ...)`.
  - Key parameters (hardcoded in `Config` class): `fs=80`, `freq_band=[8,20]`, `das_dist_start=8300`, `das_dist_end=14000`, `dx≈16.335`, time range `2024-04-24 10:20:00 → 2024-06-22 17:10:00`, `t_delta=10min`, `cpu_workers=min(10, 0.2×core_count)`.
- **Outputs**: `Output_1_SCCs_and_MAD/SCCs_h5/<template_name>/CCs_stack_<template_name>_UTC_YYYYMMDD_HHMMSS.ffffff.h5`, each h5 containing `CCs_s` (stacked normalized cross-correlation coefficient time series, 1D, 48001 points) and `RAWCCs_s` (stacked raw unnormalized cross-correlation, 1D). Also writes `config.json` under `SCCs_h5/`.
- **Core processing logic**:
  1. `preload_all_templates`: load all templates into a dict; `get_max_template_length` finds the longest template sample count.
  2. Build tasks for each 10-minute start time in `t_array`, parallelize with `ProcessPoolExecutor`; **parallelism granularity = time window**.
  3. `process_single_time_window_for_all_templates`: each window calls `Read_DAS_data` only once (duration = 10 min + longest template length), then inner-loops over all templates reusing the data.
  4. For each template: skip if output h5 already exists (supports resume); otherwise compute per-time-step per-channel `CCs_f`/`RAWCCs_f` via `CCs_comp`, trim to `expected_samples` (48001 points), take channel-wise mean to get `CCs_s`/`RAWCCs_s`, write via `write_h5`.
- **Dependent functions modules**: `readers.Read_DAS_data` / `Read_template` / `write_h5`; `compute_correlation.CCs_comp`; `pydas_readers`, `load_das_h5_PE2024` (used indirectly through readers).

### Script 2: `1_Compute_SCCs_10min_h5_Parallel.py` [Legacy / v3, ignorable]

- **Purpose**: Same functional goal as Script 1, but all paths point to `matched_filter_v3`, templates are 13 old `temp_0..temp_12`, `freq_band=[5,20]`, outputs to `SCCs_h5_6h/`; inconsistent with current release data. Each `(template, window)` task reads DAS data separately (the same window is read N times), with no output trimming. Predecessor of Script 1, now superseded. **Cannot be run to reproduce existing results in this release.**

### Script 3: `2_MAD_of_SCCs_calculation_Parallel.py` [Main, MAD Computation]

- **Purpose**: Read SCCs h5 files produced by Script 1; compute 5 statistical quantities (MAD / Median / Mean / MeanAbsDev / PeakCCs) for each template's SCCs within fixed time windows (default 1 hour) in parallel; save as pkl files and produce plots.
- **Inputs**:
  - `{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/` (Script 1 output).
  - 5 templates; key parameters: `NUM_CPUS=20`, `nt_delta_arr=[1*60]` (statistical window 60 minutes), same time range as Script 1.
- **Outputs**: Under `Output_1_SCCs_and_MAD/MAD_arrays/20240424_102000_to_20240622_171000/<t_delta>/`:
  - `MAD_array_dict.pkl`, `Median_array_dict.pkl`, `MEAN_array_dict.pkl`, `MEANAbsDev_array_dict.pkl`, `PeakCCs_array_dict.pkl` (one time series per template per time window).
  - `MAD_Average_dict.pkl` (average MAD for each template over the entire observation period — **the key file consumed by downstream Run_2**).
  - Plots: `SCCs_MAD_array_AllTemp.png`, `SCCs_Statistics_array_AllTemp_inOne.png`.
- **Core processing logic**:
  1. Build time window boundary array; outer loop over templates; inner loop builds tasks per time window; `ProcessPoolExecutor.map` calls `calculate_mad_for_chunk` in parallel (parallelism granularity = time window).
  2. `calculate_mad_for_chunk`: concatenate all 10-minute SCCs files within the window using `Read_CCs_stack_h5`; compute 5 statistics using `np.nanmedian/nanmean/nanmax`.
  3. Collect and write pkl files; compute the overall average MAD and write `MAD_Average_dict.pkl`; then generate plots.
  - Output subdirectory name is determined by `str(timedelta(minutes=nt_delta))`: 60 min → `"1:00:00"`, 720 min → `"12:00:00"`.
- **Dependent functions modules**: `readers.Read_CCs_stack_h5` / `write_pkl` / `load_pkl`; `plot.Ticks_xaxis_day_or_hour`.

### Script 4: `2_MAD_of_SCCs_calculation.py` [Legacy / v3, ignorable]

- **Purpose**: Early non-parallel version of MAD computation. Points to `matched_filter_v3`, 13 old templates, `nt_delta=[240]` (4 hours). **The actual serial computation body has been commented out entirely**; when run, it only loads existing pkl → recomputes `MAD_Average_dict.pkl` → plots. Jupytext export format with large amounts of commented-out code. Superseded by Script 3. **Does not produce the current MAD data in this release.**

---

## III. Execution Order

```
Output_0/Template_npy_files/*.npy  +  IN_DAS_data/continuous_scan_1h/
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: 1_Compute_SCCs_10min_h5_Parallel_io_once.py │  ← Main
│   writes Output_1/SCCs_h5/<template>/CCs_stack_*.h5  │
│         + SCCs_h5/config.json                        │
└─────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: 2_MAD_of_SCCs_calculation_Parallel.py       │  ← Main
│   reads SCCs_h5/  →  writes MAD_arrays/.../<window>/*.pkl │
└─────────────────────────────────────────────────────┘
                │
                ▼
   Run_2 trigger: reads SCCs_h5 (Read_CCs_stack_h5) + MAD_Average_dict.pkl (12:00:00 subdir)
   Run_3 etc.: reads SCCs_h5
```

Script 1 must complete before Script 3 (Script 3's input is Script 1's output).

---

## IV. Data Format Reference

### 4.1 SCCs h5 Files (produced by Script 1, consumed by Run_2/3 etc.)

- **Naming**: `SCCs_h5/<template_name>/CCs_stack_<template_name>_UTC_<YYYYMMDD>_<HHMMSS.ffffff>.h5`, where the UTC time in the filename is the start of that 10-minute window.
- **Contents**:
  - `CCs_s`: shape `(48001,)`, `float64`. Stacked (channel-mean) normalized cross-correlation time series; 48001 = 10 min × 60 s × 80 Hz + 1. Values approximately ±0.x; closer to 1 means more similar to the template.
  - `RAWCCs_s`: shape `(48001,)`, `float64`. Stacked raw unnormalized cross-correlation; magnitude very small (≈1e-5).
- **Read/write**: written by `functions/readers.py`'s `write_h5`; read by `Read_CCs_stack_h5` (sorts by filename timestamp, concatenates multiple 10-minute files within a specified interval into a continuous sequence).

### 4.2 MAD and Statistics pkl Files (produced by Script 3, consumed by Run_2)

- **Path**: `MAD_arrays/20240424_102000_to_20240622_171000/<t_delta>/`; in practice `<t_delta>` has two subdirectories: `1:00:00/` and `12:00:00/` (produced by runs with `nt_delta=60` and `720` respectively).
- **File structure** (all are dicts; keys include `'t_array'` and 5 template names):
  - `MAD_array_dict.pkl` / `Median_array_dict.pkl` / `MEAN_array_dict.pkl` / `MEANAbsDev_array_dict.pkl` / `PeakCCs_array_dict.pkl`: `t_array` is the window boundary array; each template key holds the corresponding statistic series for that window (shape matches the number of time windows).
  - `MAD_Average_dict.pkl`: each template key → overall-period average MAD (shape `(1,)`, `float64`). **Run_2 actually loads `12:00:00/MAD_Average_dict.pkl` as the threshold scale.**
- **Accompanying plots**: `SCCs_MAD_array_AllTemp.png`, `SCCs_Statistics_array_AllTemp_inOne.png` (produced by Script 3).

### 4.3 config.json (produced by Script 1)

- **Path**: `SCCs_h5/config.json`. Records `fs=80`, `freq_band=[8,20]`, `das_dist_start=8300`, `das_dist_end=14000`, `dx≈16.335`, 5 template names, input and output directories — fully consistent with Script 1's `Config` class.
