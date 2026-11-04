# Run_3_record_peakAmp — Documentation

## Script Classification for This Release

- **[Main]** `1_Record_Amp_Parallel.py`: Multi-process (10 workers) peak DAS amplitude extractor. Reads the triggered event list from Output_2, extracts a short DAS waveform window around each event, computes per-channel peak amplitude, and writes both per-event pkl files and the summary CSV to Output_3.
- **[Broken, ignorable]** `1_Record_Amp.py`: Old single-process version. Contains an undefined `config` object → `NameError` on run; cannot be executed. Kept in place for historical reference only.

---

## I. Folder Overview and Position in the Overall Pipeline

**Purpose**: For every triggered event (from Output_2), extract a short DAS waveform window, apply the 8–20 Hz bandpass filter, and record the per-channel peak amplitude (`amp_max`). These per-channel amplitudes serve as a proxy for local seismic intensity and are used in the optional amplitude-based magnitude branch (`Quasi_Mag_Amp`) of Run_4.

**Upstream sources**:
- Output_2 trigger pkl files: `Output_2_Triggered_SCCs_points/level_40/<template>/CCs_trig_*.pkl`.
- DAS continuous data: package-relative default `IN_DAS_data/continuous_scan_1h/` (the 80 Hz archive `DATA_DEC_80Hz_16m_SORTED/`, same path as Run_1).

**Downstream destinations**:
- Output_3 per-event pkl files: `Output_3_Trig_PeakAmp/level_40/<template>/event_<Rt_trig>.pkl`.
- Output_3 summary CSV: `Output_3_Trig_PeakAmp/level_40/Trig_MAX_amp_df.csv`.

**Pipeline position**:
```
Run_2 (trigger Output_2) + DAS continuous data (IN_DAS_data/continuous_scan_1h/)
        │
        ▼
Run_3 (this folder: peak amplitude extraction)
        │
        ▼
Output_3 → Run_4 Script 2b (optional Quasi_Mag_Amp branch only)
```

**Critical-path note**: Run_3/Output_3 are **not on the critical path for `Real_Mag`**. The final calibrated magnitude (`Real_Mag`) uses `max_RAWCCs` (from Output_2 trigger pkl) + template constants (from `Template_library.csv`). Run_4 Script 2b currently reads Output_3 unconditionally, but only uses its `amp_log_mean`/`amp_max_mean` columns for the optional `Quasi_Mag_Amp` branch; `Real_Mag` values are unaffected by Output_3.

---

## II. Script Description

### `1_Record_Amp_Parallel.py` — Main Script

- **Purpose**: Parallel extraction of per-channel peak DAS amplitudes for all triggered events.
- **Self-contained paths**: Uses the `# === dasmf path bootstrap ===` block at the top; all I/O paths written as `f"{PROJECT_ROOT}/..."`.

**Key parameters** (set in the `Config` class or at the top of the script):
```python
freq_band       = [8, 20]       # Hz
das_dist_start  = 8300          # m  (8.3 km)
das_dist_end    = 14000         # m  (14.0 km)
level           = 40            # trigger threshold level (reads level_40 from Output_2)
num_cpus        = 10            # parallel workers
template_names  = [5 templates]
window_before   = 1.5           # seconds before Rt_left
window_after    = 2.5           # seconds after Rt_right
checkpoint_every = 100          # events between checkpoint saves
```

**Processing logic**:
1. `main()` reads all Output_2 trigger pkl files for the specified level and assembles the event list.
2. Parallelizes over events using `multiprocessing.Pool(num_cpus)`.
3. Per event (`process_event`):
   - Read DAS data from `[Rt_left − window_before, Rt_right + window_after]`.
   - Apply 8–20 Hz bandpass filter (`scipy.signal.butter` + `filtfilt`).
   - Compute `amp_max[i] = max(|waveform[i, :]|)` for each channel `i` in the 8.3–14 km range (~349 channels).
   - Compute `amp_max_mean = np.mean(amp_max)` and `amp_log_mean = np.mean(np.log10(np.abs(amp_max) + 1e-12))`.
   - Write per-event pkl dict (15+ keys including `amp_max` array).
4. Every `checkpoint_every` events, saves `checkpoint.pkl` (progress state) to support resume.
5. After all events, assembles `Trig_MAX_amp_df.csv` from per-event results.

**Resume support**: If `checkpoint.pkl` exists, skips already-processed events and continues from the checkpoint. Safe to re-run after interruption.

**Outputs**:
- `Output_3_Trig_PeakAmp/level_40/config.json`.
- `Output_3_Trig_PeakAmp/level_40/<template>/event_<Rt_trig>.pkl` (one per event).
- `Output_3_Trig_PeakAmp/level_40/Trig_MAX_amp_df.csv` (summary, tab-separated, 16 columns).
- `Output_3_Trig_PeakAmp/level_40/checkpoint.pkl`.

**Dependent function modules**: `readers.Read_DAS_data` / `load_pkl` / `write_pkl`; `trigger.Trigger_with_MADarray` (for index alignment); `extract.extract_waveform` (for DAS window extraction).

---

### `1_Record_Amp.py` — Broken Legacy Script (Ignorable)

- Old single-process prototype; references an undefined `config` object → raises `NameError: name 'config' is not defined` immediately on execution.
- Uses old parameter values: `freq_band=[5,20]` (vs. `[8,20]`), 13 templates (vs. 5), `level=25` (vs. 40).
- **Cannot be run; serves only as historical reference**.

---

## III. Key Difference from the Old Script

| Parameter | Old `1_Record_Amp.py` | Current `1_Record_Amp_Parallel.py` |
|---|---|---|
| `freq_band` | [5, 20] Hz | **[8, 20] Hz** |
| Number of templates | 13 (old naming) | **5** (current naming) |
| Threshold level | 25 | **40** |
| Processing | Serial | **10-process parallel** |
| Status | Broken (NameError) | **Active** |

The 8–20 Hz band is consistent with the Run_1 cross-correlation bandpass and the noise analysis in the companion repository `WANGYuhan2002/yuhan_das_spectra` (`noise_psd/`, folder `check_minmag_pkg/noise_psd/` in `matched_filter_release.zip`).

---

## IV. Execution Requirements

1. **Output_2 must be complete** for the specified level (i.e., Run_2 Script 1 must have completed for `level_40`).
2. **DAS continuous data must be accessible** at `IN_DAS_data/continuous_scan_1h/` (place or symlink the 80 Hz archive there): the script reads raw DAS h5 files for every event's waveform window.
3. Run from the project root or any location where the `.dasmf_root` sentinel can be found.

```bash
python Run_3_record_peakAmp/1_Record_Amp_Parallel.py
# Writes to Output_3_Trig_PeakAmp/level_40/
# Checkpoints every 100 events; safe to Ctrl+C and restart
```
