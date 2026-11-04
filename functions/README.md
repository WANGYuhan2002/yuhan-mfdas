# functions/ — Documentation

## I. Overview

The `functions/` directory is the shared library for the entire `yuhan_dasmf_v1.0` MF release. It contains 14 Python modules (13 actively used + 1 dead code), plus the `pydas_readers/` sub-package for reading DAS HDF5 files.

---

## II. Module List and Classification

| Module | Status | Purpose | Called by Run_ |
|---|---|---|---|
| `load_das_h5_PE2024.py` | **Core** | DAS HDF5 custom loader | Run_1, Run_3, Run_5, Run_6 (via `readers`) |
| `pydas_readers/` | **Core** | Silixa iDASv2 HDF5 reader (sub-package) | Indirectly via `readers` |
| `compute_correlation.py` | **Core** | Per-channel cross-correlation (`CCs_comp`) and bandpass filtering | Run_0 (script 2), Run_1, Run_5 |
| `readers.py` | **Core** | All I/O helpers (pkl/h5 read-write, DAS loading, CCs h5 loading) | Every Run_ |
| `trigger.py` | Active | MAD-based trigger detection (`Trigger_with_MADarray`) | Run_2, Run_3 |
| `classify.py` | Active | Overlap de-duplication and event classification (pandas version) | Run_4 |
| `stalta.py` | Active (partially) | `sta_lta_centred_unnorm` on SCCs; other functions are dead code | Run_2, Run_5 |
| `extract.py` | Active | DAS waveform extraction around trigger times | Run_3, Run_4 (scripts 3a, 5_) |
| `extract_station_info.py` | Active | NOA IMS catalog parser | Run_0, Run_4, Run_6 |
| `cluster.py` | Active | DBSCAN event clustering | Run_0 (script 3) |
| `plot.py` | Active | Time-axis tick formatters and DAS waveform waterfall plotting | Every Run_ (plotting scripts) |
| `noa.py` | Active (Run_6 only) | FDSN station metadata download (via `requests`), requires network | Run_6 (script 4) |
| `das_mffi_io.py` | Active (Run_0 only) | Read MFFI external DAS data; data folder fixed to the package-relative `IN_MFFI_data/kefalonia_ts_h5_80Hz/` (MFFI data not distributed) | Run_0 (scripts 1, 3_1) |
| `append_cut_convert.py` | **Dead code** | Early prototype for segment cutting; not imported by any main script | — |

### `pydas_readers/` Sub-package

- A standalone Python package for reading Silixa iDASv2 / downsampled `.h5` DAS files. Used by `readers.py` for reading DAS data.
- `mapping/channel_mapping.py`: `get_mapping()` would read a channel-mapping table (`mapping/Channel_mapping_information_catalouge_v4.1.csv`, channel index → latitude / longitude / distance along fiber). That table belongs to a different DAS deployment and is **not included**; `get_mapping()` is not called anywhere in this release, so no script needs it.

---

## III. Detailed Module Descriptions

### `readers.py` — All I/O Functions

The most widely used module; imported by every Run_ step.

**Key functions**:

| Function | I/O | Notes |
|---|---|---|
| `load_pkl(path)` | reads pkl | `pickle.load` with `encoding='latin1'` (compatible with Python 2 pickles) |
| `write_pkl(data, path)` | writes pkl | `pickle.dump` with `protocol=4` |
| `write_h5(data, key, path)` | writes h5 | writes one dataset under one key |
| `load_h5(key, path)` | reads h5 | reads one dataset |
| `Read_template(template_name, template_dir)` | reads `.npy` | loads raw template waveform array |
| `Read_DAS_data(t_start, t_end, fs, f1, f2, dx, d1, d2)` | reads DAS h5 | wraps `load_das_custom` + `pydas_readers`; path config at lines 116–117 |
| `Read_CCs_stack_h5(template, h5_dir, t_start, t_end)` | reads CCs h5 | sorts h5 files by filename timestamp, concatenates within time range; **`MESSUP_Read_CCs_stack_h5` is deprecated (unsorted, buggy)** |
| `Read_CCs_trig_pkl(template, pkl_dir, ...)` | reads trigger pkl | yields per-pkl dict |
| `Read_CCs_trig_pkl_toPANDAS(...)` | reads trigger pkl → DataFrame | aggregates all pkl files across all templates into a single DataFrame |

**Default DAS path**: lines 116–117 set the default 80 Hz DAS directory (`das_h5_mnt_80Hz`) to `f'{PROJECT_ROOT}/IN_DAS_data/continuous_scan_1h/'` (and the unused 400 Hz path `das_h5_mnt_raw` to `f'{PROJECT_ROOT}/IN_DAS_data/DATA_RAW_400Hz_8m/'`). Place or symlink the continuous DAS archive there.

---

### `compute_correlation.py` — Cross-Correlation Engine

**Key function**: `CCs_comp(template, data, fs, freq_band, ...)` → `(CCs_f, RAWCCs_f)`:
- Bandpass-filters both template and data in `freq_band`.
- Computes normalized cross-correlation (NCC) per channel via FFT.
- Returns `CCs_f` (normalized, ≈ −1 to 1) and `RAWCCs_f` (unnormalized, in physical units) as 2D arrays `[time, channels]`.
- Used by Run_1 (per-10-min window) and Run_5 (f-k analysis window).

---

### `trigger.py` — MAD-Based Trigger Detection

**Key function**: `Trigger_with_MADarray(CCs_s, MAD_array, t_array, level, min_interval, ...)`:
- Applies threshold = `MAD_array[ii] × level` piecewise (one MAD value per time segment).
- Detects trigger peaks above threshold; enforces minimum event spacing `min_interval`.
- Returns trigger times, SCCs values, left/right boundaries, and pulse widths.
- Used by Run_2 (main trigger step) and Run_3 (for indexing).

---

### `classify.py` — Overlap De-duplication and Classification

**Active (pandas) version**:

| Function | Purpose |
|---|---|
| `Remove_Temp_Offest_forSeries(df, Template_library_csv)` | corrects template-specific time offsets in trigger times |
| `Read_Collapse_group_df(pkl_dir, Template_library_csv, ...)` | reads all trigger pkl files into a single DataFrame with all trigger fields |
| `Collapse_with_MAX_CCs(df, time_window_s)` | de-duplicates overlapping detections (from multiple templates) within `time_window_s`; keeps the detection with the highest `|CCs_s_trig|` |

**Old (dict-based) version**: several functions with `_dict` suffix in the module — deprecated, superseded by the pandas version above. Not called by any current main script.

---

### `stalta.py` — STA/LTA on SCCs

**The only truly called function in the MF release**:

`sta_lta_centred_unnorm(signal, sta_len, lta_len)`:
- Computes centred (causal + anti-causal symmetric) unnormalized STA/LTA ratio on the input 1D `signal` (the SCCs time series, not raw waveforms).
- Used in Run_2 to compute `max_stalta` (the STA/LTA peak value at each trigger point, used as a pulse-sharpness / quality criterion).
- Also used in Run_5 to compute STA/LTA of f-k-filtered CCs.

**Dead code in this file** (not called by any main script in the MF release):
- `sta_lta_centred_multichan(...)`: multi-channel STA/LTA — dead.
- `trig_onechan_and_outindex(...)`: per-channel trigger — dead.
- `trig_allchan_and_collapsed_index(...)`: multi-channel collapsed trigger — dead.
- The former `sys.path.append(...)` to the external `stalta_clustering/` project has been removed (see the comment at lines 24–25); STA/LTA catalogues now come from `IN_STALTA_cata/`.

---

### `extract_station_info.py` — NOA IMS Catalog Parser

**Key function**: `extract_station_info(file_path, station_name)`:
- Parses an IMS 1.0 format NOA catalog file.
- Filters rows to the specified station (e.g., `'FSK'` for Fiskardo, Kefalonia).
- Returns a list of dicts, each with: `EQ_time`, `EQ_latitude`, `EQ_longitude`, `EQ_depth`, `EQ_magnitude`, `STA_name`, `STA_distance` (**in degrees**, not km; multiply by 111 for km), `STA_azimuth`, `STA_phase`, `STA_arrival_time`.

---

### `cluster.py` — Event Clustering

**Key function**: `cluster_events(matched_df, ...)`:
- Applies DBSCAN with distance metric `D = 1 − |R|` (1 minus absolute cross-correlation coefficient), `eps=0.90`, `min_samples=2`.
- Groups similar events into clusters for template candidate selection.
- Used only in Run_0 (Script 3).
- **Note**: line 75 contains a string equality check on the DAS data path — if the path string changes, the check silently fails.

---

### `plot.py` — Plotting Utilities

Used by all Run_ plotting scripts. Key functions include:
- `Ticks_xaxis_day_or_hour(ax, ...)` / `Ticks_xaxis_minute_or_second(ax, ...)`: time-axis tick formatters.
- `waterfall_date(ax, data, ...)`: DAS waveform waterfall (time × channel) plotting with date-labeled x-axis.
- `generate_gs_grid(...)`: custom GridSpec layout helper.

---

### `noa.py` — NOA Station Codes and FDSN Station Metadata Download

- Extracts station codes from an IMS 1.0 file and downloads station coordinates from the NOA EIDA FDSN server (`http://eida.gein.noa.gr/fdsnws/station/1/query`) via `requests`.
- Requires an active network connection and `requests` / `pandas` / `geopandas` / `cartopy` / `geopy` (does not use `obspy`).
- Only used by Run_6, Script 4 (`4_plot_detection_power_mag_dist_map*.py`); not needed for the main catalog pipeline.

---

### `das_mffi_io.py` — MFFI External Data Reader

- Reads DAS data from the MFFI (Multichannel F-K Imaging) analysis directory.
- MFFI data path is the package-relative `IN_MFFI_data/kefalonia_ts_h5_80Hz/` (external data, not distributed with this release). The DAS input folder is compared with the package-relative defaults `IN_DAS_data/continuous_scan_1h/` (80 Hz), `IN_DAS_data/DATA_DEC_8Hz_16m_SORTED/` and `IN_DAS_data/DATA_RAW_400Hz_8m/` to choose the reader and channel spacing, so pass exactly these folders.
- Only used by Run_0 (Script 1 for STA/LTA catalog reading and Script 3_1 for paper figures).
- Not needed for the main template matching pipeline.

---

## IV. Pipeline Usage by Run Step

| Run step | Modules used |
|---|---|
| Run_0 | `extract_station_info`, `cluster`, `stalta` (indirectly, via `readers`), `compute_correlation`, `das_mffi_io`, `plot` |
| Run_1 | `compute_correlation.CCs_comp`, `readers` (Read_DAS_data, Write_h5), `plot` |
| Run_2 | `readers.Read_CCs_stack_h5`, `trigger.Trigger_with_MADarray`, `stalta.sta_lta_centred_unnorm`, `plot` |
| Run_3 | `readers`, `trigger`, `extract`, `plot` |
| Run_4 | `classify.Read_Collapse_group_df` + `Collapse_with_MAX_CCs`, `trigger`, `readers`, `extract_station_info`, `plot` |
| Run_5 | `compute_correlation`, `stalta.sta_lta_centred_unnorm`, `trigger`, `classify`, `plot` |
| Run_6 | `extract_station_info`, `noa`, `plot` |

---

## V. Python Path Bootstrap

All scripts in this release locate `functions/` via a `.dasmf_root` sentinel file search. Each script contains:

```python
# === dasmf path bootstrap ===
import sys, pathlib
_f = pathlib.Path(__file__).resolve()
for _p in [_f, *_f.parents]:
    if (_p / '.dasmf_root').exists():
        PROJECT_ROOT = str(_p); break
sys.path.insert(0, str(pathlib.Path(PROJECT_ROOT) / 'functions'))
```

This means no absolute paths are needed; the package works wherever it is placed, as long as `.dasmf_root` exists at the repository root.
