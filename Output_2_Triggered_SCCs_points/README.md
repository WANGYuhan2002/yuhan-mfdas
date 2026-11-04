# Output_2_Triggered_SCCs_points — Documentation

> **GitHub version:** the data files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_2_Triggered_SCCs_points/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> This folder is a **pure data folder (no scripts)**. It holds the output of the "MAD×level threshold triggering on stacked cross-correlation SCCs" step in the template matching pipeline. All triggering in this release uses the kurtosis version (output pkl files include the `max_stalta` key).

## Script Classification for This Release

This folder itself contains no scripts. The scripts that produce the data are in `../Run_2_trigger_SCCs_kurtosis/`:

- **[Main]** `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres_Parallel.py`: Triggers on MAD×level threshold and computes STA/LTA on SCCs to record `max_stalta`; reads Output_1's SCCs_h5 + MAD_Average_dict.pkl as inputs, writes to this folder.
- **[Legacy (ignorable)]** `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres.py`: Historical v3 non-parallel version, no `max_stalta`, does not produce data in this folder.
- **[Plotting (ignorable)]** `2_Show_Trig_TimeHisto_SCCsPoints.py`: Reads pkl files from this folder to produce plots; depends on the STA/LTA catalogue `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative copy of the external STA/LTA project's catalogue; external data, in the Zenodo archive, not in the GitHub repository).

---

## I. Folder Overview and Position in the Overall Pipeline

**Purpose:** Store trigger points. For each template, these are candidate detections where `|SCCs(t)|` exceeds the `MAD × Threshold_level` threshold on the SCCs time series; clusters are reduced to local maxima and recorded together with CC values, raw cross-correlation, pulse width, boundary times, and the STA/LTA peak on SCCs (`max_stalta`). Used downstream for de-overlap, classification, and false-positive removal.

**Position in the overall pipeline:**

```
Output_1_SCCs_and_MAD/SCCs_h5/      (upstream: continuous SCCs, one h5 per 10 minutes)
Output_1_SCCs_and_MAD/MAD_arrays/   (upstream: per-template MAD threshold baseline)
            │
            ▼
   Run_2_trigger_SCCs_kurtosis/
   1_Trigger_SCCs_eachTemp_byMADandLevel_constThres_Parallel.py
            │
            ▼
 ===> This folder: Output_2_Triggered_SCCs_points/level_XX/template_*/CCs_trig_*.pkl <===
            │
            ├──► Run_3_record_peakAmp/     (record DAS peak amplitudes at trigger times)
            ├──► Run_4_assign_overlaps_with_kurtosis/  (de-overlap, classify, estimate magnitudes; strongly depends on max_stalta)
            └──► Run_5_fk_exclude_false_positive/      (f-k analysis to exclude false positives)
```

---

## II. Directory Structure and Naming Conventions

```
Output_2_Triggered_SCCs_points/
├── level_15/   level_20/   level_25/   level_30/   level_35/   level_40/   level_50/
│   ├── config.json                         ← parameter snapshot for this level (written by Run_2)
│   ├── template_0423_80Hz/
│   ├── template_0425_80Hz/
│   ├── template_10_80Hz/
│   ├── template_20_80Hz/
│   └── template_8_80Hz/
│       └── CCs_trig_<template>_UTC_<YYYYMMDD>_<HHMMSS.ffffff>.pkl
```

- **`level_XX`**: `XX = Threshold_level`, i.e. the MAD multiple for triggering. Trigger criterion: `|SCCs(t)| > MAD_template × Threshold_level` (see `functions/trigger.py::Trigger_with_MADarray`). Seven levels exist: `level_15/20/25/30/35/40/50`, produced by a parameter sweep; higher level = stricter = fewer triggers.
- **`template_*_80Hz`**: 5 template subdirectories, one per template (`template_0423/0425/10/20/8`, 80 Hz sampling rate).
- **Filename `CCs_trig_<template>_UTC_<time>.pkl`**: `<time>` is the UTC start of that 10-minute slice (format `YYYYMMDD_HHMMSS.ffffff`). A file covers `[t_start, t_start+10min)`; **pkl files are only written when there is at least one trigger point in the slice** (empty slices are not saved).
- **Time range:** `2024-04-24 10:20:00 → 2024-06-22 17:10:00` (approximately 59 days).
- **Scale:** approximately 55,379 pkl files in total across all folders, approximately 223 MB.

---

## III. Internal Structure of Each pkl

Each `pkl` is a `dict` with 10 keys; all arrays have the same length (= number of trigger points N in that 10-minute slice):

| Field | Type / Shape | Meaning and Units |
| --- | --- | --- |
| `t_trig` | ndarray[datetime], (N,) | Trigger-point time (the sample with max `\|SCCs\|` within the cluster, UTC) |
| `CCs_s_trig` | ndarray[float64], (N,) | SCCs value at trigger point (signed, approx. [−1, 1]) |
| `RAWCCs_s_trig` | ndarray[float64], (N,) | Raw cross-correlation value at trigger point (unnormalized; used downstream for magnitude estimation) |
| `width_trig` | ndarray[timedelta], (N,) | Time width of the trigger cluster (`t_right − t_left`) |
| `t_left` | ndarray[datetime], (N,) | Left boundary of trigger cluster (first above-threshold sample) |
| `t_right` | ndarray[datetime], (N,) | Right boundary of trigger cluster (last above-threshold sample) |
| `SCCs_of_MAD` | ndarray[float64], (N,) | `CCs_s_trig / MAD_template`; absolute value ≥ `Threshold_level` for a trigger to fire |
| `max_RAWCCs` | list[float], (N,) | Max `\|RAWCCs\|` (signed) within the `[t_left, t_right]` window; NaN for empty window |
| `t_arg_max` | list[datetime], (N,) | Time corresponding to `max_RAWCCs` |
| `max_stalta` | list[float], (N,) | Max STA/LTA (unnormalized) of SCCs within a `search_win=[0.6, 0.2]s` window around the trigger point; used by downstream Run_4 for false-positive filtering; NaN for empty window |

`max_stalta` is the core additional field of the kurtosis version. Downstream `functions/readers.py::Read_CCs_trig_pkl` depends on this key; Run_4 and Run_5 both require this field to exist.

---

## IV. config.json (one per level subdirectory)

Written by the Run_2 main-line script at runtime; records the complete parameters used to produce the data for that level:

| Field | Meaning |
| --- | --- |
| `Threshold_level` | MAD multiple threshold for this level (15/20/.../50, matches directory name) |
| `Minimal_event_interval` | Minimum interval between events in the same cluster (seconds) |
| `IN_CCs_stack_dir` | Input SCCs source directory |
| `MAD_Avg_path` | Input MAD baseline path |
| `OUTPUT_dir` | Output directory (i.e. the level subdirectory of this folder) |
| `fs` | Sampling rate: 80 Hz |
| `Temp_name_list` | The 5 templates used in triggering |
| `sta_len / lta_len` | Short-window / long-window lengths (seconds) for STA/LTA on SCCs: 0.5 / 1.5 |
| `search_win` | Left and right window (seconds) for searching STA/LTA peak at each trigger point: [0.6, 0.2] |

---

## V. Downstream Consumption

`functions/readers.py::Read_CCs_trig_pkl(IN_CCs_trig_dir, Template, t_start, t_end)`:
1. Globs all `*.pkl` files in the `template_*` subdirectory.
2. Parses the 10-minute slice start time from each filename; filters by time window.
3. Concatenates all fields (including `max_stalta`) and trims by `t_trig` before returning.

Downstream steps select which level directory to consume via the `Threshold_level` parameter:

- **Run_3**: records DAS peak amplitudes at trigger times.
- **Run_4**: de-overlap, classify, estimate magnitudes; uses `max_stalta > threshold` to filter false positives.
- **Run_5**: f-k analysis to exclude false positives; also reads `max_stalta`.
