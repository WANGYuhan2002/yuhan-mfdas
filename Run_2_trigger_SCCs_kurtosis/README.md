# Run_2_trigger_SCCs_kurtosis — Documentation

> All triggering in this release uses the kurtosis version (output pkl files include the `max_stalta` key).

## Script Classification for This Release

- **[Main]** `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres_Parallel.py`: Triggers on MAD×level threshold and computes STA/LTA on SCCs to record `max_stalta`; reads Output_1's SCCs_h5 + MAD_Average_dict.pkl as inputs; writes to Output_2.
- **[Legacy (ignorable, kept in place)]** `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres.py`: Historical v3 non-parallel version; no STA/LTA logic; old template naming; does not produce the current Output_2 data.
- **[Plotting (ignorable)]** `2_Show_Trig_TimeHisto_SCCsPoints.py`: Trigger-point time histograms/scatter plots; reads the STA/LTA reference catalogue from `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative; a copy of the external STA/LTA project's catalogue — external data, in the Zenodo archive, not in the GitHub repository); also depends on the external package `pydas_readers`.

---

## I. Folder Overview and Position in the Overall Pipeline

**Upstream sources:**
- `Output_1_SCCs_and_MAD/SCCs_h5/` (Run_1 output): per-template continuous stacked cross-correlation SCCs (`CCs_s`) and raw cross-correlation RAWCCs (`RAWCCs_s`), 10-minute slice h5 files. Read via `functions/readers.py::Read_CCs_stack_h5`.
- `Output_1_SCCs_and_MAD/MAD_arrays/.../12:00:00/MAD_Average_dict.pkl`: per-template average MAD dict (12-hour window), the trigger threshold baseline.

**Outputs from this folder:**
- `../Output_2_Triggered_SCCs_points/level_{N}/{template}/CCs_trig_{template}_UTC_{YYYYMMDD_HHMMSS.ffffff}.pkl`: trigger-point dict (containing `max_stalta`).
- `../Output_2_Triggered_SCCs_points/level_{N}/config.json`: run parameters (including `sta_len/lta_len/search_win`).
- `../Output_2_Triggered_SCCs_points/show_trigger/level_{N}/...png`: trigger-point histograms/scatter plots (produced by the plotting script).

**Downstream consumers (of `Output_2_Triggered_SCCs_points`):**
- `Run_3_record_peakAmp/`: extracts DAS peak amplitudes at trigger times.
- `Run_4_assign_overlaps_with_kurtosis/`: de-overlap, classify, estimate magnitudes (strongly depends on `max_stalta`).
- `Run_5_fk_exclude_false_positive/`: f-k analysis to exclude false positives (also reads `max_stalta`).

**Position in the overall pipeline:**
`Run_1` → **`Run_2 (this step: triggering + STA/LTA features)`** → `Run_3` → `Run_4` → `Run_5` → `Run_6`.

---

## II. Script-by-Script Description

### Script 1: `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres_Parallel.py` (parallel version, main line)

- **Purpose:** Multi-process triggering on each template's 10-minute SCCs slices using the `MAD×level` threshold; compute the STA/LTA maximum `max_stalta` on the SCCs at each trigger point; write pkl files.
- **Self-contained paths:** The script has a `# === dasmf path bootstrap ===` block at the top; all I/O paths are written as `f"{PROJECT_ROOT}/..."`.
- **Inputs:**
  - SCCs/RAWCCs: `{PROJECT_ROOT}/Output_1_SCCs_and_MAD/SCCs_h5/` → `Read_CCs_stack_h5`.
  - MAD: `{PROJECT_ROOT}/Output_1_SCCs_and_MAD/MAD_arrays/.../12:00:00/MAD_Average_dict.pkl` → `load_pkl`.
  - Key parameters: `Threshold_level=50`, `Minimal_event_interval=3s`, `fs=80`, 5 templates `template_0423/0425/10/8/20_80Hz`, time range `2024-04-24 10:20 ~ 2024-06-22 17:10`, `t_delta=10min`, `num_cpus=10`, `sta_len=0.5s`, `lta_len=1.5s`, `search_win=[0.6,0.2]s`.
- **Outputs:** `{PROJECT_ROOT}/Output_2_Triggered_SCCs_points/level_50/{template}/CCs_trig_*.pkl` + `config.json` for that level.
- **Core processing logic:**
  1. `main()` creates directories, saves `config.json`; outer loop iterates over templates; `Pool.imap_unordered` parallelizes `process_segment`.
  2. `process_segment`: if pkl already exists, skip and return the existing trigger count.
  3. **Wide-window read:** reads `[t_start - lta_len, t_end + sta_len]` of `CCs_s_wide, RAWCCs_s_wide`, then trims back to `[start_index:end_index]` to get `CCs_s, RAWCCs_s` (avoids STA/LTA transients at slice boundaries).
  4. **Compute STA/LTA:** calls `sta_lta_centred_unnorm(·, sta_len*fs, lta_len*fs)` on the entire `CCs_s_wide` segment, then trims and aligns to get `sl_unnorm`.
  5. **Trigger:** `Trigger_with_MADarray(...)` yields `CCs_s_trig, t_trig, t_left, t_right, width_trig` (threshold = MAD×level; neighboring events within `Minimal_event_interval` are removed).
  6. Compute sample index `idx_t_trig`; filter out-of-bounds; extract `RAWCCs_s_trig`.
  7. For each trigger point's `[t_left,t_right]` window: take `max |RAWCCs|` → `max_RAWCCs/t_arg_max` (NaN for empty window).
  8. **Compute `max_stalta`:** for each trigger point, take the max of `sl_unnorm` within `[idx-0.6·fs, idx+0.2·fs]`; NaN for empty window.
  9. Compute `SCCs_of_MAD = CCs_s_trig / MAD`; assemble `OUT_dict` (10 keys) → `write_pkl`.
- **Output pkl fields (10 keys):** `t_trig, CCs_s_trig, RAWCCs_s_trig, width_trig, t_left, t_right, SCCs_of_MAD, max_RAWCCs, t_arg_max, max_stalta`.
- **Dependent functions modules:** `readers.Read_CCs_stack_h5 / write_pkl / load_pkl`, `trigger.Trigger_with_MADarray`, `stalta.sta_lta_centred_unnorm`.
- **Note:** "kurtosis" is a historical colloquial label; the criterion implemented in code is the STA/LTA ratio on SCCs, not the mathematical fourth-moment kurtosis. `constThres` means a piecewise constant threshold: `Trigger_with_MADarray` segments by MAD time grid; each segment's threshold = `MAD_grid[ii] × Threshold_level` (constant within each segment).

### Script 2: `1_Trigger_SCCs_eachTemp_byMADandLevel_constThres.py` (non-parallel, legacy / ignorable)

- Serial nested-loop early prototype; all paths hardcoded to an older project directory; `Threshold_level=40`; MAD uses `4:00:00` window; template list uses old naming `temp_0…temp_12` (13 templates); `freq_band=[5,20]`.
- **Has no STA/LTA / `max_stalta` logic whatsoever**; output pkl fields are minimal (9 keys); is NOT the producer of the current Output_2 data.
- Historical legacy, kept in place, does not need to be run.

### Script 3: `2_Show_Trig_TimeHisto_SCCsPoints.py` (plotting, ignorable)

- Reads already-triggered pkl files; plots per-template event time histograms (bin=1h) and all-template trigger-point scatter overlaid with an external STA/LTA directory reference.
- **External data dependency:** the STA/LTA reference catalogue is read from `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative; originally `stalta_clustering/Output_1_STALTA_catalog_v1/...` in the external STA/LTA project); the file must be present before running (it is in the Zenodo archive, not in the GitHub repository). Also depends on `pydas_readers` (external package) and `load_das_h5_PE2024`.
- **Inputs:** trigger pkl directory → `Read_CCs_trig_pkl`; `Template_library.csv`; external `sta_lta_catalog.csv`; MAD dict (for plotting threshold lines).
- **Outputs:** `Output_2_Triggered_SCCs_points/show_trigger/level_XX/...png` (histogram and trigger-point scatter plots).
- Pure plotting script; does not produce the pkl data in Output_2.

---

## III. Execution Order and Dependencies

```
Output_1_SCCs_and_MAD/{SCCs_h5, MAD_arrays}
        │
        ▼
[Script 1: 1_..._Parallel.py]   ← Main: MAD×level trigger + compute max_stalta
   reads SCCs_h5 (wide window) + MAD_Average_dict.pkl → STA/LTA on SCCs
        │  writes (10-key pkl)
        ▼
   ../Output_2_Triggered_SCCs_points/level_{N}/{template}/CCs_trig_*.pkl + config.json
        │
        ├──►[Script 3: 2_Show...py]  reads pkl → plots (show_trigger/), optional visual check
        │
        └──► Downstream: Run_3 → Run_4 (strongly depends on max_stalta) → Run_5 → Run_6
```

Output_1 must exist before running Script 1 (main line). Script 3 is for visual inspection only and has no hard prerequisite relationship with downstream steps.

---

## IV. Data Notes

Scripts in this folder write to `../Output_2_Triggered_SCCs_points/` (see that folder's documentation). All data currently on disk was produced by the parallel main-line script in this folder.

**Core fields of the output pkl:**

| Key | Meaning |
|---|---|
| `t_trig` | Trigger time (UTC) |
| `CCs_s_trig` | SCCs value at trigger point (approx. −1 to 1) |
| `RAWCCs_s_trig` | Raw cross-correlation value at trigger point (unnormalized) |
| `width_trig` | Trigger pulse width (timedelta) |
| `t_left / t_right` | Left/right boundary times of the trigger window |
| `SCCs_of_MAD` | `CCs_s_trig / MAD`; absolute value ≥ level triggers |
| `max_RAWCCs` | Max `\|RAWCCs\|` (signed) within `[t_left, t_right]` window; NaN for empty window |
| `t_arg_max` | Time corresponding to `max_RAWCCs` |
| `max_stalta` | **Core field**: max STA/LTA ratio on SCCs within `search_win` at the trigger point (pulse sharpness); NaN for empty window |

`max_stalta` is the core additional field of this variant. Downstream Run_4 strongly depends on this key for false-positive filtering.
