# Run_0_preparing_templates_from_catalog — Documentation

> **GitHub version:** the STA/LTA snapshot and result sub-folders of this folder described below (e.g. `threshold_100_trigNO_100_sta0.3s_lta3s/`) are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Run_0_preparing_templates_from_catalog/`). "Bundled" below refers to that archive. The NOA bulletin `IN_NOA_cata/100km_distance/...` is not redistributed (link only, see `../IN_NOA_cata/README.md`).

## Script Classification for This Release

- **[Main, Script 1]** `1_SNR_Mag_control_find_NOAeq_SatisfyStalta_OverlapUseTime.py`: Matches NOA catalog events against the STA/LTA trigger catalog to find template candidates with high SNR.
- **[Main, Script 2]** `2_Comp_corr_between_events_v6.py`: Computes pairwise DAS cross-correlation between template candidates → produces `correlation_matrix.npy` and `offset_matrix.npy`.
- **[Main, Script 3]** `3_Similarity_control_delete_similar_eqs.py`: DBSCAN clustering on correlation matrix; keeps the highest-magnitude event per cluster (+ isolated events) → produces `dup_temp_df.csv` and `corr_matrix_dup_temp.npy`.
- **[Paper figure, ignorable]** `3_1_Similarity_control_delete_similar_eqs_plot_for_paper.py`: Paper-figure variant of Script 3 (`DBSCAN_clustering_and_map_summary.png`); uses the same local `functions/` modules via the path bootstrap and does not write `dup_temp_df.csv`.
- **[Main, Script 4]** `4_Produce_templates_from_EQs.py`: Template waveform overview figure from `dup_temp_df.csv` (`save_as_temp_or_not=0` in this release → does NOT write `.npy` files here; template `.npy` waveforms and final `Template_library.csv` columns are produced in `Output_0_Template_library/`).

---

## I. Folder Overview and Position in the Overall Pipeline

This folder is **Step 0** of the template matching pipeline — it selects, validates, and saves the 5 reference earthquake waveforms (templates) from which the matched-filter scan proceeds.

**Upstream sources**:
- **STA/LTA trigger catalog** (local, bundled): `threshold_100_trigNO_100_sta0.3s_lta3s/sta_lta_catalog.csv` within this folder (~194 KB, already in release; not external).
- **NOA 100km catalog**: `IN_NOA_cata/100km_distance/IMS_20240425_20240623_319.txt` (also bundled).
- **DAS continuous data**: the 80 Hz archive `DATA_DEC_80Hz_16m_SORTED/`, read from the package-relative default `IN_DAS_data/continuous_scan_1h/` (place or symlink the archive there; external, required for Scripts 2 and 3).

**Downstream destinations**:
- `Output_0_Template_library/Template_npy_files/*.npy` — produced by Script 3; consumed by Run_1.
- `Output_0_Template_library/Template_library.csv` — overwritten by Script 3; consumed by Run_4 (magnitude calibration) and Run_6.
- `Output_0_Template_library/correlation_matrix.npy` + `offset_matrix.npy` — produced by Script 2; consumed by Script 3.

---

## II. Active Parameter Set

Multiple sub-directories exist in this folder reflecting different parameter combinations explored during development. The **active / formal version** is:

```
threshold_100_trigNO_100_sta0.3s_lta3s/
└── Output_1_Find_NOAeq_Mag2.0_10sec_SNR30_100km/
    └── matched_df.csv     ← template candidate list for this parameter set
```

Parameters for this formal version:
- STA/LTA: `sta=0.3s`, `lta=3s`, trigger threshold = 100.
- NOA match: `mag_cut=2.0`, search window = ±10 s.
- SNR cut: `SNR_cut=30`, `peak_amp_cut≥900`.
- Distance radius: 100 km.

Other sub-directories (`threshold_200_...`, `threshold_25_...`, etc.) are historical explorations. Their `matched_df.csv` files and clustering results reflect different parameter choices and **should not be used** with the current release pipeline.

---

## III. Script-by-Script Description

### Script 1: `1_SNR_Mag_control_find_NOAeq_SatisfyStalta_OverlapUseTime.py`

- **Purpose**: Time-match NOA earthquake events (ML ≥ `mag_cut`) against STA/LTA triggers; further filter by SNR and peak amplitude to get high-quality template candidates.
- **Inputs**:
  - `sta_lta_catalog.csv` (local, bundled): columns `Unnamed:0, peak_time, peak_amp, peak_score, peak_dist, max_amp_mean`. `peak_amp ≈ SNR²`; `peak_score` = number of triggered channels.
  - NOA 100km IMS catalog: `extract_station_info(file, 'FSK')` → list of dicts with `EQ_time`, `EQ_magnitude`, etc.
- **Filters applied**:
  - NOA magnitude: `EQ_magnitude ≥ mag_cut = 2.0`.
  - Time match: NOA `EQ_time` within ±10 s of `STA_arrival_time` → candidate triggers.
  - SNR: `sqrt(peak_amp) ≥ SNR_cut = 30` (i.e., `peak_amp ≥ 900`).
  - Peak amplitude: `peak_amp ≥ peak_amp_cut = 900`.
- **Output**: `matched_df.csv` — table of matched NOA events + STA/LTA trigger metadata; template candidate list.

---

### Script 2: `2_Comp_corr_between_events_v6.py`

- **Purpose**: Read DAS data for each candidate event; compute pairwise cross-correlation between all candidates; save correlation and offset matrices.
- **Inputs**:
  - `matched_df.csv` (Script 1 output): `N` candidate events.
  - DAS continuous data (`IN_DAS_data/continuous_scan_1h/`).
- **Parameters**: `freq_band=[8,20]`, `das_dist_start=8300`, `das_dist_end=14000`.
- **Core computation**: for each pair `(i, j)`, extract DAS windows, bandpass filter, compute cross-correlation, find peak → `R[i,j]` and `offset[i,j]`.
- **Outputs**:
  - `correlation_matrix.npy`: shape `(N, N)`, float32.
  - `offset_matrix.npy`: shape `(N, N)`, int32 (sample offsets at peak correlation).

---

### Script 3: `3_Similarity_control_delete_similar_eqs.py`

- **Purpose**: Apply DBSCAN to cluster similar template candidates; select the largest-magnitude event per cluster (+ isolated events) as deduplicated candidates (`dup_temp_df.csv`).
- **Inputs**:
  - `matched_df.csv` (Script 1 output).
  - `correlation_matrix.npy` (Script 2 output).
- **DBSCAN parameters**: distance metric `D = 1 − |R|`, `eps=0.90`, `min_samples=2`.
- **Key outputs** (under `Output_3_Delete_similar_eqs/`):
  - `dup_temp_df.csv`: deduplicated candidate template table.
  - `corr_matrix_dup_temp.npy`: sub-correlation matrix of deduplicated candidates.
  - `matched_df_with_clusters.csv`: full candidate table with cluster labels.
  - `cluster_labels.npy`: DBSCAN cluster label array.
- **Note**: Final `.npy` waveform extraction and overwriting `Template_library.csv` with SNR/amplitude columns are performed by the scripts in `Output_0_Template_library/` (`2_check_again_map_Produce_templates_from_EQs.py` and `3_add_SNR_and_Energy_peakAmp_for_Template_library.py`).

---

### Script 3_1: `3_1_Similarity_control_delete_similar_eqs_plot_for_paper.py` (Paper figure, Ignorable)

- Paper-figure variant of Script 3 (`DBSCAN_clustering_and_map_summary.png`); uses the same local `functions/` modules via the path bootstrap and does not write `dup_temp_df.csv`.

---

### Script 4: `4_Produce_templates_from_EQs.py`

- Visualization of candidate template waveforms (`ALL_templates_waterfall_map.png`).
- Key parameter `save_as_temp_or_not = 0` → does **not** write `.npy` files; only generates figures.
- Template `.npy` files are produced in `Output_0_Template_library/` (`2_check_again_map_Produce_templates_from_EQs.py`).

---

## IV. Execution Order

```
[Bundled] sta_lta_catalog.csv  +  IN_NOA_cata/100km/IMS_*.txt
                  │
                  ▼
     Script 1 (1_SNR_Mag_control_...): match NOA vs STA/LTA → matched_df.csv
                  │
     IN_DAS_data/continuous_scan_1h/ ──┐
                  │                    ▼
                  └────► Script 2 (2_Comp_corr_between_events_v6.py): pairwise DAS cross-correlation
                              → correlation_matrix.npy + offset_matrix.npy
                                        │
                              ▼
                    Script 3 (3_Similarity_control_delete_similar_eqs.py): DBSCAN cluster + deduplicate
                              → dup_temp_df.csv + corr_matrix_dup_temp.npy
                                        │
                                        ▼
                    Script 4 (4_Produce_templates_from_EQs.py): overview figure
                                        │
                                        ▼
                    Output_0_Template_library (scripts 1 → 2 → 3: final 5 templates *.npy + 27-col CSV)
                                        │
                                        ▼
                         Run_1 (SCCs computation)
```

Scripts 1 → 2 → 3 → 4 run in order. DAS continuous data (`IN_DAS_data/continuous_scan_1h/`) is required for Scripts 2 and 4.

---

## V. Historical Sub-directories

Several sub-directories in this folder reflect earlier parameter explorations:

| Sub-directory | Notes |
|---|---|
| `threshold_100_trigNO_100_sta0.3s_lta3s/` | **Active / formal version** |
| `threshold_200_...` | Higher threshold — fewer templates |
| `threshold_25_...` | Lower threshold — more candidates |
| Other variants | Different `lta`/`sta` ratios or time windows |

All non-active sub-directories can be ignored. Their `matched_df.csv` files are not compatible with the current 5-template release.
