# Output_1_SCCs_and_MAD — Documentation

> **GitHub version:** the data files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_1_SCCs_and_MAD/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> Release path: `yuhan_dasmf_v1.0/Output_1_SCCs_and_MAD/`
> All contents produced by scripts in `Run_1_compute_SCCs_and_MAD/`.

---

## I. Directory Role

This directory stores the two key intermediate products of Step 1 (Run_1):

1. **SCCs_h5/**: Per-template 10-minute Stacked Cross-Correlation Coefficient time-series files — the raw detection signal for the entire observation period.
2. **MAD_arrays/**: Per-template Median Absolute Deviation (MAD) statistics computed over the SCCs, used as the noise baseline for trigger thresholding in Step 2 (Run_2).

---

## II. Directory Structure

```
Output_1_SCCs_and_MAD/
├── SCCs_h5/
│   ├── config.json                          # Run_1 configuration record
│   ├── template_0423_80Hz/
│   │   ├── CCs_stack_template_0423_80Hz_UTC_20240424_102000.000000.h5
│   │   ├── CCs_stack_template_0423_80Hz_UTC_20240424_103000.000000.h5
│   │   └── ...  (144 files for 2024-04-25 only; full run: ~8537 files per template)
│   ├── template_0425_80Hz/ ...
│   ├── template_10_80Hz/  ...
│   ├── template_20_80Hz/  ...
│   └── template_8_80Hz/   ...
└── MAD_arrays/
    └── 20240424_102000_to_20240622_171000/
        ├── 1:00:00/
        │   ├── MAD_array_dict.pkl
        │   ├── Median_array_dict.pkl
        │   ├── MEAN_array_dict.pkl
        │   ├── MEANAbsDev_array_dict.pkl
        │   ├── PeakCCs_array_dict.pkl
        │   └── MAD_Average_dict.pkl
        └── 12:00:00/
            ├── MAD_array_dict.pkl
            ├── ...
            └── MAD_Average_dict.pkl     ← ★ consumed by Run_2 for trigger thresholds
```

---

## III. Data Bundling Status

**Partially bundled**: Only the 2024-04-25 data is included in the release as a sample dataset.

| Sub-directory | Bundled content | Full-run size |
|---|---|---|
| `SCCs_h5/<template>/` | 144 h5 files per template (one day: 2024-04-25, 144 × 10-min windows) | ~8537 files per template for the full 59-day period |
| `MAD_arrays/` | **Complete** — full observation period (2024-04-24 10:20 → 2024-06-22 17:10) | Already complete |

To run the full pipeline beyond the sample day, SCCs h5 files for the remaining period must be recomputed by running Run_1 with the continuous DAS archive placed at `IN_DAS_data/continuous_scan_1h/`.

---

## IV. Data Format: SCCs h5 Files

**Naming convention**:
```
CCs_stack_<template_name>_UTC_<YYYYMMDD>_<HHMMSS.ffffff>.h5
```
The UTC timestamp in the filename is the **start of that 10-minute window**.

**Contents of each h5 file**:

| Dataset | Shape | Dtype | Description |
|---|---|---|---|
| `CCs_s` | `(48001,)` | float64 | Stacked (channel-mean) **normalized** cross-correlation time series. 48001 = 10 min × 60 s × 80 Hz + 1. Values ≈ ±0.01 (small positive/negative correlation; values near ±1 indicate strong template match) |
| `RAWCCs_s` | `(48001,)` | float64 | Stacked (channel-mean) **raw unnormalized** cross-correlation. Physical units; magnitude ≈ 1e−5 (DAS strain-rate units) |

**Read function**: `functions/readers.py`'s `Read_CCs_stack_h5(template_name, h5_dir, t_start, t_end)`:
- Sorts h5 files by filename timestamp.
- Selects files whose time window overlaps `[t_start, t_end]`.
- Concatenates selected files into a single continuous array.
- Returns `(CCs_s_concat, RAWCCs_s_concat, t_array_concat)`.

---

## V. Data Format: MAD Arrays pkl Files

**Path**: `MAD_arrays/20240424_102000_to_20240622_171000/<window>/`

Two `<window>` subdirectories exist:
- `1:00:00/`: MAD computed in 1-hour windows.
- `12:00:00/`: MAD computed in 12-hour windows. **This is the one consumed by Run_2.**

**File contents** (all are Python dicts with keys `'t_array'` and one key per template name):

| File | Per-template value | Shape / Notes |
|---|---|---|
| `MAD_array_dict.pkl` | Time series of MAD values, one per window | `(n_windows,)` |
| `Median_array_dict.pkl` | Time series of median SCCs per window | `(n_windows,)` |
| `MEAN_array_dict.pkl` | Time series of mean SCCs per window | `(n_windows,)` |
| `MEANAbsDev_array_dict.pkl` | Time series of mean absolute deviation per window | `(n_windows,)` |
| `PeakCCs_array_dict.pkl` | Time series of peak SCCs per window | `(n_windows,)` |
| `MAD_Average_dict.pkl` | **Scalar** overall-period average MAD for each template | shape `(1,)` float64; **Run_2 reads this** |

**How Run_2 uses MAD**:
```python
MAD_Average = load_pkl(f'{MAD_dir}/12:00:00/MAD_Average_dict.pkl')
threshold = MAD_Average[template_name] * Threshold_level  # e.g. level=50
```
Events where `|CCs_s| > threshold` are triggered.

---

## VI. config.json

Path: `SCCs_h5/config.json`

Records the Run_1 parameters used to produce the SCCs:

```json
{
  "fs": 80,
  "freq_band": [8, 20],
  "das_dist_start": 8300,
  "das_dist_end": 14000,
  "dx": 16.335,
  "template_names": ["template_0423_80Hz", "template_0425_80Hz", "template_10_80Hz", "template_20_80Hz", "template_8_80Hz"],
  "input_dir": "...",
  "output_dir": "..."
}
```

`das_dist_start=8300` m and `das_dist_end=14000` m define the DAS channel segment used for stacking (approximately 8.3–14 km fiber distance, ~367 channels). The 8–20 Hz bandpass is applied before cross-correlation.
