# Output_0_Template_library — Documentation

> **GitHub version:** the data files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_0_Template_library/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> Release path: `yuhan_dasmf_v1.0/Output_0_Template_library/`
> All contents produced by scripts in `Run_0_preparing_templates_from_catalog/`.

---

## I. Directory Role

This directory is the **template library** — the starting point of the entire MF pipeline. It stores the 5 selected reference earthquake waveforms (templates) and their associated metadata. Downstream Run_1 reads these files to compute Stacked Cross-Correlation Coefficients (SCCs) against continuous DAS data.

---

## II. Directory Structure

```
Output_0_Template_library/
├── Template_library.csv            # ★ Core metadata table (5 rows × 27 columns)
├── Template_npy_files/             # ★ Core template waveforms (raw unfiltered DAS arrays)
│   ├── template_0423_80Hz.npy      # K2, ML2.5 — shape (641, 936)
│   ├── template_0425_80Hz.npy      # I2, ML3.0 — shape (881, 936)
│   ├── template_10_80Hz.npy        # K1, ML2.4 — shape (1121, 936)
│   ├── template_20_80Hz.npy        # I1, ML3.1 — shape (721, 936)
│   └── template_8_80Hz.npy         # K3, ML4.1 — shape (961, 936)
├── correlation_matrix.npy          # (5×5) float32 — inter-template correlation
├── offset_matrix.npy               # (5×5) int32 — inter-template time offsets (samples)
└── [historical files, ignorable]
    ├── matched_df_with_clusters.csv
    ├── dup_temp_df_SNR30.csv
    ├── dup_temp_df_SNR40.csv
    └── Template_library_K16_I13_F14.csv
```

---

## III. Template Overview

The 5 templates are selected reference earthquakes from the DAS deployment period, chosen for their high signal-to-noise ratio and spatial coverage within the Kefalonia seismogenic zone.

| Template name (file) | Plot label | NOA ML | Location | Notes |
|---|---|---|---|---|
| `template_10_80Hz` | K1 | ML 2.4 | Kefalonia fault, south segment | Longest waveform (1121 samples) |
| `template_0423_80Hz` | K2 | ML 2.5 | Kefalonia fault, north segment | |
| `template_8_80Hz` | K3 | ML 4.1 | Kefalonia fault, central | Largest magnitude template |
| `template_20_80Hz` | I1 | ML 3.1 | Ionian Sea | |
| `template_0425_80Hz` | I2 | ML 3.0 | Ionian Sea | |

K1/K2/K3 are on the Kefalonia Transform Fault; I1/I2 are in the broader Ionian Sea region. Together they cover both main seismotectonic environments sampled by the DAS cable.

---

## IV. File Descriptions

### `Template_library.csv` (5 rows × 27 columns)

The master metadata table for all templates. Key columns:

| Column | Meaning |
|---|---|
| `infile_name` | Template `.npy` filename (without path) |
| `plot_name` | Short label for figures (K1, K2, K3, I1, I2) |
| `plot_order` | Display order in multi-template plots |
| `t_P_sec` | P-wave arrival time within the template waveform (seconds from array start) |
| `t_S_sec` | S-wave arrival time within the template waveform (seconds from array start) |
| `EQ_latitude` | Template earthquake hypocenter latitude (degrees) |
| `EQ_longitude` | Template earthquake hypocenter longitude (degrees) |
| `self_CC` | **Self cross-correlation amplitude** — `max |CCs| / normalization`; a template constant used in magnitude calibration (`fac_CC2amp = amp_max_mean / self_CC`) |
| `amp_max_mean` | **Mean of per-channel peak amplitudes** (DAS strain rate, 8.3–14 km segment, 8–20 Hz filtered); template constant used in magnitude calibration |
| `amp_log_mean` | Mean of `log10(|amp_max| + 1e-12)` per channel |
| `SNR_S` | S-wave signal-to-noise ratio (dB) |
| `SNR_P` | P-wave signal-to-noise ratio (dB) |

**Important**: Script 3 in this directory (`3_add_SNR_and_Energy_peakAmp_for_Template_library.py`) **overwrites** `Template_library.csv` by appending computed columns (`SNR_S`, `SNR_P`, `self_CC`, `amp_max_mean`, `amp_log_mean`). It must be run after Script 2 (`2_check_again_map_Produce_templates_from_EQs.py`) has produced `Template_npy_files/*.npy`.

---

### `Template_npy_files/*.npy` — Raw Template Waveforms

- **Format**: `numpy.ndarray`, shape `(n_samples, 936)`, dtype `float64`.
- **Channels**: 936 total DAS channels (full cable, 0–15.3 km fiber, 80 Hz sampling).
- **Waveform content**: raw unfiltered strain-rate DAS waveforms at 80 Hz. Filtering (8–20 Hz bandpass) is applied on-the-fly during cross-correlation in Run_1 (`compute_correlation.CCs_comp`).
- **Duration**:

| Template | n_samples | Duration (s) |
|---|---|---|
| template_10_80Hz (K1) | 1121 | ~14.0 s |
| template_8_80Hz (K3) | 961 | ~12.0 s |
| template_0425_80Hz (I2) | 881 | ~11.0 s |
| template_20_80Hz (I1) | 721 | ~9.0 s |
| template_0423_80Hz (K2) | 641 | ~8.0 s |

The total file size of `Template_npy_files/` is approximately 55 MB; this is bundled in the release.

---

### `correlation_matrix.npy` and `offset_matrix.npy`

- `correlation_matrix.npy`: shape `(5, 5)`, float32. The mutual cross-correlation between each pair of the 5 templates; used by Run_0 Script 3 to cluster similar templates.
- `offset_matrix.npy`: shape `(5, 5)`, int32. The sample-lag offset at the peak cross-correlation between each template pair.
- Both produced by Run_0 Script 2; consumed by Run_0 Script 3 for DBSCAN clustering.

---

### Historical Files (Ignorable)

These files reflect earlier parameter explorations and are kept in place for provenance but are not used by any active pipeline script:

| File | Origin |
|---|---|
| `matched_df_with_clusters.csv` | Early template candidate table including DBSCAN cluster labels |
| `dup_temp_df_SNR30.csv` | Template candidates filtered at SNR ≥ 30 threshold |
| `dup_temp_df_SNR40.csv` | Template candidates filtered at SNR ≥ 40 threshold |
| `Template_library_K16_I13_F14.csv` | Early candidate pool (43 templates) before final 5 were selected |

---

## V. How Templates Are Used Downstream

| Downstream step | Files consumed | Purpose |
|---|---|---|
| Run_1 (`1_Compute_SCCs_10min_h5_Parallel_io_once.py`) | `Template_npy_files/*.npy` (all 5) | Compute per-channel NCC between template and continuous DAS; stack across channels → SCCs time series |
| Run_4 (`2b_Est_Mag_with_GR_law_check.py`) | `Template_library.csv` → `self_CC`, `amp_max_mean` | Compute `fac_CC2amp = amp_max_mean / self_CC`; magnitude formula: `Quasi_Mag = log10(|max_RAWCCs × fac_CC2amp|)` |
| Run_6 (`4_plot_detection_power_mag_dist_map.py`) | `Template_library.csv` → lat/lon, plot_name | Plot template locations on map; label detections by template |

---

## VI. Scripts in This Directory and Execution Order

Unlike the other `Output_*` folders, this directory also contains the 4 finalization / re-check scripts that operate on the 5 curated templates in `Template_library.csv` (after `Run_0` candidate selection):

- **[Main, Script 1]** `1_check_again_Comp_corr_between_events_v6.py`: Recomputes the 5×5 inter-template cross-correlation (`correlation_matrix.npy`) and time-offset (`offset_matrix.npy`) matrices in `Check_again_corr_map_waterfall/`.
- **[Main, Script 2]** `2_check_again_map_Produce_templates_from_EQs.py`: Cuts the raw unfiltered DAS waveforms (`save_as_temp_or_not=1`) to produce `Template_npy_files/*.npy`, plus single-template and overview figures.
- **[Main, Script 3]** `3_add_SNR_and_Energy_peakAmp_for_Template_library.py`: Appends `SNR_S`, `SNR_P`, `self_CC`, `amp_max_mean`, and `amp_log_mean` to `Template_library.csv` (overwriting it with the final 27-column version).
- **[Paper figure, ignorable]** `4_check_again_map_and_template_waterfall_plot_for_paper.py`: Paper-figure script (`save_as_temp_or_not=0`, does not write `.npy` files). Appends `external/GRL_DAS_MF_plots/functions/` at the end of `sys.path` as a fallback, while the modules it imports (`plot`, `load_das_h5_PE2024`, `compute_correlation`) resolve from this repository's `functions/` first.

```
Run_0 ( scripts 1 → 2 → 3 → 4: candidate selection & DBSCAN deduplication )
    ↓
1_check_again_Comp_corr_between_events_v6.py → correlation_matrix.npy + offset_matrix.npy
    ↓
2_check_again_map_Produce_templates_from_EQs.py → Template_npy_files/*.npy
    ↓
3_add_SNR_and_Energy_peakAmp_for_Template_library.py → overwrites Template_library.csv (27 cols)
    ↓
4_check_again_map_and_template_waterfall_plot_for_paper.py (optional paper figure)
```

Scripts 1 → 2 → 3 in this directory must run in order. Script 2 produces the `.npy` template waveform files, and Script 3 overwrites `Template_library.csv` with the 5 computed SNR/amplitude columns.
