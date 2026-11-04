# Run_5_fk_exclude_false_positive — Documentation

## Script Classification for This Release

- **[Main, Script 1]** `1_CCs_fk_for_trig_points.py`: For each triggered event, computes the f-k (frequency-wavenumber) / f-s (frequency-slowness) amplitude spectrum and produces multi-panel diagnostic PNG figures. Used to visually identify false positives (e.g., traffic noise with characteristic slowness).
- **[Exploratory, no disk output]** `2_CCs_ZLCC_for_trig_points.py`: Computes zero-lag cross-correlation and group similarity metrics. Output directory `Output_5_CCs_fk/2_CCs_ZLCC/` does **not exist on disk** — this script produced no results in the current run; ignore.
- **[Paper figure]** `3_CCs_kurtosis_example_for_paper.py`: Produces paper-quality figures contrasting high-quality (`max_stalta` ∈ [20,1000]) vs. low-quality (`max_stalta` ∈ [1,10]) events.

**Auxiliary modules** (in this folder):
- `spectrum.py`: `Calculate_fk_amplitude_spectrum(signal_2d, dt, dx)` — core f-k computation.
- `group_similarity.py`: Group similarity metrics (used only by Script 2).

---

## I. Folder Overview and Position in the Overall Pipeline

**Purpose**: Apply frequency-wavenumber / frequency-slowness analysis to triggered DAS events to diagnose false positives. DAS records all wave types propagating along the fiber — earthquakes produce wavefields at characteristic apparent velocities, while noise (traffic, marine, anthropogenic) may appear at very different slownesses and can thereby be identified and excluded.

**Upstream sources**:
- Output_2 trigger pkl files (event list, trigger times, `max_stalta` values).
- Output_0 template library (`Template_library.csv`, `Template_npy_files/*.npy`) and Output_1 SCCs/MAD.
- DAS continuous data (package-relative default `IN_DAS_data/continuous_scan_1h/`, for waveform extraction).

**Outputs** (in `Output_5_CCs_fk/`):
- Script 1: per-event PNG figures in `Output_5_CCs_fk/1_CCs_fk_spec/`.
- Script 2: no disk output (directory `Output_5_CCs_fk/2_CCs_ZLCC/` does not exist).
- Script 3: paper figures in `Output_5_CCs_fk/3_CCs_kurtosis_example_for_paper/`.

**Pipeline position**:
```
Output_2 (triggers) + Output_0/1 (templates, SCCs, MAD) + DAS data (IN_DAS_data/continuous_scan_1h/)
        │
        ▼
Script 1: f-k / f-s spectrum → multi-panel diagnostic PNGs (visual inspection)
        │  (no automatic threshold; manual review)
        ▼
Output_5_CCs_fk/1_CCs_fk_spec/ (per-event PNGs)
```

This step provides visual evidence for excluding false positives but does **not** produce a filtered CSV or modify the event catalog. The exclusion decisions are applied manually by reviewing the generated figures.

---

## II. Script-by-Script Description

### Script 1: `1_CCs_fk_for_trig_points.py` (Main)

- **Purpose**: For triggered events falling within the selected correlation and `max_stalta` ranges, recompute the per-channel cross-correlation matrix `CCs_f`, compute its 2D f-s amplitude spectrum, and save a multi-panel diagnostic figure.
- **Inputs**:
  - Output_2 trigger pkl files (`Output_2_Triggered_SCCs_points/level_20/`, aggregated via `Read_Collapse_group_df`), plus `Template_library.csv`, `Template_npy_files/*.npy`, and `Output_1_SCCs_and_MAD/`.
  - DAS continuous data (package-relative default `IN_DAS_data/continuous_scan_1h/`; same path as Run_1/3).
  - Key parameters: `freq_band=[8,20]`, `das_dist_start=8300`, `das_dist_end=14000`, `Search_win=[0.5, 0.5]`, `kurtosis_range=[1, 12]`, `C_trig_range=[0.1, 1]`.
- **Core computation** (via `spectrum.py`'s `Calculate_fk_amplitude_spectrum`):
  1. Extract DAS window (time × channels) and compute `CCs_f` via `CCs_comp`.
  2. Apply 2D FFT: one-sided in frequency (positive frequencies), fftshifted in wavenumber.
  3. Convert wavenumber axis to slowness: `s = k / f` (s/m).
  4. Output: `(freqs, wavenums, ampspec)` — note the code comments call this "f-k" but the x-axis plotted is **f-s (frequency-slowness)**, not f-k.
- **Outputs**: PNG named `<plot_name>_<|CCs_s_trig|>_<CCs_s_trig_ofMAD>_<YYYYMMDD_HHMMSS_ffffff>.png` saved to `Output_5_CCs_fk/1_CCs_fk_spec/<dir_byPara>/` (plus `config.json` in `Output_5_CCs_fk/1_CCs_fk_spec/`).
- **Figure content**: multi-panel layout — waveform waterfall, f-s spectrum amplitude, SCCs time series with trigger marker, STA/LTA curve, and `CCs_f` waterfall.
- **No automatic threshold**: there is no numerical criterion for automatic false-positive classification — the f-s figures are reviewed visually. Events with anomalous slowness (e.g., apparent velocity < 1 km/s indicating traffic or surface waves) can be manually flagged.

---

### Script 2: `2_CCs_ZLCC_for_trig_points.py` (Exploratory, No Output)

- Computes zero-lag cross-correlation (ZLCC) between events and group similarity metrics (using `group_similarity.py`).
- Output directory `Output_5_CCs_fk/2_CCs_ZLCC/` **does not exist on disk** — no results were saved during the current project run. This script produced no output. Ignore.

---

### Script 3: `3_CCs_kurtosis_example_for_paper.py` (Paper Figure)

- Generates publication-quality multi-panel figures contrasting high-quality events (`max_stalta` ∈ [20, 1000]) and low-quality events (`max_stalta` ∈ [1, 10]).
- `kurtosis_range` parameter:
  - High-quality examples: `kurtosis_range=[20, 1000]`.
  - Low-quality examples: `kurtosis_range=[1, 10]`.
- ("Kurtosis" is the historical label for `max_stalta` in this codebase.)
- Output: `Output_5_CCs_fk/3_CCs_kurtosis_example_for_paper/<dir_byPara>/*.png` (paper-quality figures).
- Reads from the same upstream trigger / template / SCCs / DAS inputs as Script 1.

---

## III. Auxiliary Modules

### `spectrum.py`

**Key function**: `Calculate_fk_amplitude_spectrum(signal_2d, dt, dx)`

- **Input**: `signal_2d` — 2D NumPy array, shape `(n_time, n_channels)`.
  - `dt`: time sampling interval (s); `dx`: channel spacing (m).
- **Processing**:
  1. 2D FFT (`np.fft.rfft2` or equivalent).
  2. Frequency axis: one-sided (positive only), range 0 to `1/(2dt)`.
  3. Wavenumber axis: `fftshift` → range −`1/(2dx)` to `+1/(2dx)`.
- **Return**: `(freqs, wavenums, ampspec)` — frequency array (Hz), wavenumber array (cycles/m), amplitude spectrum (abs value, shape `(n_freq, n_wavenum)`).
- **Note**: although named "f-k spectrum" throughout the code, the analysis plot produced by Script 1 uses the **f-s (frequency-slowness)** representation (x-axis = slowness, not wavenumber) to better visualize apparent velocities.

---

### `group_similarity.py`

Contains functions for computing group similarity metrics between DAS waveforms:
- `outlier_masscenter(...)`: center of mass excluding outliers.
- `loneliest_neighborhood(...)`: finds the most isolated event in waveform space.
- `pairwise_quantile(...)`: quantile of pairwise distances.
- `average_pairwise(...)`: mean pairwise cross-correlation.
- `compute_moving_similarity_metrics(...)`: computes all four metrics in a sliding time window.

These functions are only called by Script 2 (no disk output exists). Not needed for the main pipeline.

---

## IV. Output Directory

```
Output_5_CCs_fk/
├── 1_CCs_fk_spec/
│   ├── config.json
│   └── kur1_12_C0.1_1_win0.5_0.5/
│       ├── Temp_K1_0.1136_50.3_20240511_135123_025000.png
│       └── ...
└── 3_CCs_kurtosis_example_for_paper/
    ├── config.json
    ├── kur20_1000_C0.03_1_win1.5_1.5/
    └── kur1_10_C0.03_0.1_win1.5_1.5/
```

`Output_5_CCs_fk/2_CCs_ZLCC/` does **not** exist (Script 2 produced no output).

**PNG filename format** for Script 1 output:
```
<plot_name>_<|CCs_s_trig|>_<CCs_s_trig_ofMAD>_<YYYYMMDD_HHMMSS_ffffff>.png
```
where `plot_name` is the template short label (e.g., `K1`, `K2`), `|CCs_s_trig|` is the absolute SCCs value (used for visual sorting), and `CCs_s_trig_ofMAD` is the normalized SCCs/MAD value.
