# Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF — Documentation

## Script Classification for This Release

**Independent pairs (Scripts 1&2 can run in parallel; Scripts 3&4 can run in parallel after both 1&2)**:

- **[Main, Script 1]** `1_label_STALTA_with_template_plus_NOAeq.py`: STA/LTA catalog as base table; time-matches against NOA and MF detections → `stalta_catalog_enriched.csv`.
- **[Main, Script 2]** `2_label_NOA_with_template_and_detectByMForNot.py`: NOA catalog as base table; labels each event with nearest template and whether MF detected it → `noa_catalog_labeled.csv`.
- **[Main, Script 3]** `3_plot_catalog_MF_stalta_NOA.py`: Comprehensive 5-template × 5-panel comparison figure using both CSV files.
- **[Main, Script 4]** `4_plot_detection_power_mag_dist_map.py`: Magnitude–distance detection power figure with map (vertical layout); requires network (FDSN + contextily).
- **[Alternate layout, Script 4']** `4_plot_detection_power_mag_dist_map_CHANGE_LAYOUT.py`: Same data as Script 4, horizontal layout — larger map, focused extent, different symbol colors.

---

## I. Folder Overview and Position in the Overall Pipeline

This folder is the **final step** of the MF release pipeline. It performs a three-catalog cross-comparison:
1. **MF detections** (from Output_4 `Events_classified_Real_Mag.csv`).
2. **STA/LTA detections** (from external `sta_lta_catalog.csv` — external data dependency; see below).
3. **NOA official catalog** (from `IN_NOA_cata/200km_distance/IMS_*.txt`).

**Upstream inputs**:
| Data | Path |
|---|---|
| MF final catalog | `Output_4_.../level_20/MagRegression/Events_classified_Real_Mag.csv` |
| NOA 200km catalog | `IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt` |
| STA/LTA catalog | `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative copy of the external project's `stalta_clustering/Output_1_STALTA_catalog_v1/.../sta_lta_catalog.csv`) |
| Template library | `Output_0_Template_library/Template_library.csv` |
| Inset figures | `Output_6_Compare_MF_stalta_NOA/insert_fig.png` and `insert_fig_flat.png` (manually prepared) |

**External data dependency**: Script 1 reads the STA/LTA catalog from `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative; a copy of the external STA/LTA project's catalogue, in the Zenodo archive but not in the GitHub repository); Scripts 3 and 4 read the enriched copy written by Script 1. The file must be present before running Scripts 1, 3, and 4 (Script 2 does not need STA/LTA data).

**Outputs** (in `Output_6_Compare_MF_stalta_NOA/level_20/`):
- `eq_list_to_plot/stalta_catalog_enriched.csv` (Script 1 → consumed by Scripts 3/4).
- `eq_list_to_plot/noa_catalog_labeled.csv` (Script 2 → consumed by Scripts 3/4).
- `compare_figures/3_catalog_comparison_plot.png` (Script 3).
- `compare_figures/4_detection_power.png` (Script 4, vertical).
- `compare_figures/4_detection_power_flat_layout.png` (Script 4', horizontal).

---

## II. Script-by-Script Description

### Script 1: `1_label_STALTA_with_template_plus_NOAeq.py`

**Purpose**: Starting from the STA/LTA trigger catalog, label each STA/LTA event with the nearest NOA earthquake (within 3 s) and nearest MF detection (within 2 s).

**Inputs**:
- `IN_STALTA_cata/sta_lta_catalog.csv` (package-relative copy of the external STA/LTA project's catalog; in the Zenodo archive): v1 format, columns `Unnamed:0, peak_time, peak_amp, peak_score, peak_dist, max_amp_mean`.
- NOA 200km IMS catalog (via `extract_station_info`, deduplicated by `STA_arrival_time`).
- MF catalog `Events_classified_Real_Mag.csv` (filtered to `Belong_or_not==1`).
- Time window filter: 10:20–17:10 UTC (matching window) for Scripts 1/2; note Scripts 3/4 plot a slightly different window 10:15–16:15 (display choice).

**Processing**:
1. Load STA/LTA triggers; filter to the observation time window (~7251 events remain).
2. For each STA/LTA trigger, find the nearest NOA event within 3 s → add `noa_match_flag`, `noa_time_diff_s`, `noa_magnitude`, `noa_latitude`, `noa_longitude`, `noa_distance_km` (STA_distance × 111 km).
3. For each STA/LTA trigger, find the nearest MF detection within 2 s → add `mf_match_flag`, `mf_time_diff_s`, `mf_template_name`, `mf_real_mag`, `mf_sccs`.
4. Write `stalta_catalog_enriched.csv`.

**Output**: `eq_list_to_plot/stalta_catalog_enriched.csv` (~7251 rows, 17 columns; ~514 NOA matches, ~4353 MF matches).

---

### Script 2: `2_label_NOA_with_template_and_detectByMForNot.py`

**Purpose**: Starting from the NOA catalog, label each earthquake with the nearest template region and whether MF detected it.

**Inputs**:
- NOA 200km IMS catalog → `extract_station_info(file, 'FSK')`; deduplicated by `EQ_time` (~893 unique events after time-window filter).
- MF catalog (for detection label).
- `Template_library.csv` (template hypocenters for spatial proximity).
- Requires `geopy.distance.geodesic` for spatial distance computation.

**Processing**:
1. For each NOA event, compute geodesic distance to each template hypocenter; assign `template_region` = name of template within 8 km, or `'outside_all_templates'`.
2. For each NOA event, check if MF detected it: time tolerance 3 s AND spatial distance ≤ 7 km from template hypocenter → `is_detected_by_mf` (True/False).
3. Write `noa_catalog_labeled.csv`.

**Output**: `eq_list_to_plot/noa_catalog_labeled.csv` (~893 rows; ~251 detected by MF).

---

### Script 3: `3_plot_catalog_MF_stalta_NOA.py`

**Purpose**: Multi-panel comprehensive comparison figure across all 5 templates.

**Inputs**: both enriched CSVs (Scripts 1 and 2).

**Figure layout**: `figsize=(55, 11×6)`, `dpi=150`. One row per template (K1/K2/K3/I1/I2) + one summary row = 6 rows total. Each row has 5 columns:
1. Time–magnitude scatter (MF=red, STA/LTA=green, NOA=blue).
2. Event count per 2-hour time bin (stacked histogram).
3. Frequency–magnitude distribution (FMD).
4. G-R cumulative frequency (log scale).
5. Additional panel (detection rate or magnitude comparison).

**Output**: `compare_figures/3_catalog_comparison_plot.png`.

---

### Script 4: `4_plot_detection_power_mag_dist_map.py` — Magnitude–Distance Detection Power Figure (Vertical Layout)

**Purpose**: Show detection capability as a function of NOA earthquake magnitude and epicentral distance to the DAS cable.

**Inputs**: `noa_catalog_labeled.csv` + `stalta_catalog_enriched.csv` + `Template_library.csv` + inset figure `insert_fig.png`.

**Requirements**: network access for `contextily` (basemap tiles) and NOA FDSN (`noa.py`, via `requests`) for station metadata download.

**Figure layout** (vertical, 20×21 inches, dpi=250):
- Top row (full width): magnitude–distance scatter plot (NOA events, symbol size ∝ number of channels, color = detected / undetected by MF; STA/LTA events in white stars).
- Bottom-left: inset figure `insert_fig.png` (manually prepared).
- Bottom-right: map of study area, extent `lat[36.5,40.5] lon[18.9,22.7]`.

**Output**: `compare_figures/4_detection_power.png`.

---

### Script 4': `4_plot_detection_power_mag_dist_map_CHANGE_LAYOUT.py` — Horizontal Layout Variant

Same data and computation as Script 4, different panel arrangement:

| Property | Script 4 (vertical) | Script 4' (horizontal) |
|---|---|---|
| Figure size | 20×21 inches | 30×21 inches |
| Layout | Mag–dist top; inset+map bottom | Map left full height; mag–dist top-right; inset bottom-right |
| Map extent | `lon[18.9,22.7] lat[36.5,40.5]` | `lon[19.95,22.2] lat[36.7,39.5]` (focused) |
| Inset file | `insert_fig.png` | `insert_fig_flat.png` |
| STA/LTA map markers | White stars | Blue stars |
| `scaling_factor` | 30 | 70 |

**Output**: `compare_figures/4_detection_power_flat_layout.png`.

---

## III. Execution Order and Dependencies

```
[Scripts 1 and 2 can run in parallel]

Script 1: STA/LTA(external) + NOA(200km) + MF(Output_4) → stalta_catalog_enriched.csv
Script 2: NOA(200km) + MF(Output_4) + Template_library.csv → noa_catalog_labeled.csv

         ↓ [both must complete before Scripts 3/4]

Script 3: both CSVs → 3_catalog_comparison_plot.png
Script 4: both CSVs + insert_fig.png + network → 4_detection_power.png
Script 4': both CSVs + insert_fig_flat.png + network → 4_detection_power_flat_layout.png
```

**Time window micro-differences** (careful when interpreting figures):
- Scripts 1/2 match over 10:20–17:10 UTC.
- Script 3 plots 10:15–16:15 UTC (slightly narrower display window).
- Script 4 uses the full CSV with no additional time filter.

---

## IV. STA/LTA Path Dependency

The STA/LTA catalog (`IN_STALTA_cata/sta_lta_catalog.csv`, read by Script 1; Scripts 3 and 4 use its enriched copy in `Output_6_.../eq_list_to_plot/`) is **external data** (a copy of the STA/LTA project's catalogue) and must be present at runtime. Script 2 does **not** require STA/LTA data and can be run independently.

If the STA/LTA project directory is not available:
- Scripts 1, 3, 4 will fail.
- Script 2 will still run successfully.
- Output 6 CSV and figure (Script 2 outputs, and plots that don't need STA/LTA) remain producible.
