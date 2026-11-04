# Output_6_Compare_MF_stalta_NOA — Documentation

> **GitHub version:** the data and figure files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_6_Compare_MF_stalta_NOA/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> Release path: `yuhan_dasmf_v1.0/Output_6_Compare_MF_stalta_NOA/`
> This directory is a pure data/figure output directory with no scripts. All contents are produced by scripts in `Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF/`.

---

## I. Directory Role

This directory is the **final evaluation/comparison output directory** of the entire MF pipeline. It stores the results of aligning and cross-labeling three earthquake catalogs (MF / STA/LTA / NOA official), as well as detection capability comparison figures. Produced by Run_6; not read by any other Run_ step; this is the pipeline endpoint.

**Upstream (input sources, consumed by Run_6 scripts before being written to this directory)**:

| Data | Path | Produced By |
|------|------|--------|
| MF final earthquake catalog | `{PROJECT_ROOT}/Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis/level_20/MagRegression/Events_classified_Real_Mag.csv` | Run_4 series |
| NOA official catalog (IMS 1.0) | `{PROJECT_ROOT}/IN_NOA_cata/200km_distance/IMS_20240425_20240623_263.txt` | External input (Greece NOA) |
| STA/LTA catalog | `IN_STALTA_cata/sta_lta_catalog.csv` (copy of `stalta_clustering/.../sta_lta_catalog.csv`) | Independent STA/LTA project (external data dependency; in the Zenodo archive, not in the GitHub repository) |
| Template library | `{PROJECT_ROOT}/Output_0_Template_library/Template_library.csv` | Run_0 |
| Manually prepared inset figures | `insert_fig.png` and `insert_fig_flat.png` (in this directory's root) | Manually prepared, not script-generated; must be included |

---

## II. Directory Structure

```
Output_6_Compare_MF_stalta_NOA/
├── insert_fig.png                         # Manually prepared inset figure for Script 4 (original layout)
├── insert_fig_flat.png                    # Manually prepared inset figure for Script 4 (horizontal layout)
└── level_20/
    ├── eq_list_to_plot/
    │   ├── stalta_catalog_enriched.csv    # Produced by Script 1; consumed by Scripts 3/4
    │   └── noa_catalog_labeled.csv        # Produced by Script 2; consumed by Scripts 3/4
    └── compare_figures/
        ├── 3_catalog_comparison_plot.png  # Produced by Script 3
        ├── 4_detection_power.png          # Produced by Script 4 (original vertical layout)
        ├── 4_detection_power_flat_layout.png # Produced by Script 4 (horizontal layout)
        └── 4_detection_power_1111.png     # No script outputs this filename; likely a manually saved debug version
```

`level_20` corresponds to `level=20` (MF detection threshold level, consistent with the `level` parameter from upstream Run_4).

---

## III. CSV Field Descriptions

### stalta_catalog_enriched.csv (tab-separated, ~7251 rows)

**Produced by**: Run_6 Script 1 (`1_label_STALTA_with_template_plus_NOAeq.py`).
**Consumed by**: Run_6 Scripts 3, 4.

Base source: STA/LTA raw trigger catalog; approximately 7251 events after time-window filtering.

**Original columns (from `sta_lta_catalog.csv`)**:

| Column | Meaning | Units/Notes |
|------|------|-----------|
| `Unnamed: 0` | Original row index | Integer |
| `peak_time` | STA/LTA trigger peak time | UTC |
| `peak_amp` | Trigger peak amplitude | Original DAS strain rate |
| `peak_score` | STA/LTA ratio score | Dimensionless |
| `peak_dist` | Trigger peak spatial offset | All zeros in file |
| `max_amp_mean` | Mean amplitude | — |

**Columns appended by Script 1 (nearest-neighbor time matching)**:

| Column | Meaning | Units/Notes |
|------|------|-----------|
| `noa_match_flag` | Whether a NOA catalog event was matched within 3 s | `True`/`False`; ~514 True across the full file |
| `noa_time_diff_s` | Time difference to nearest NOA earthquake | Seconds; NaN if unmatched |
| `noa_magnitude` | Magnitude (ML) of matched NOA event | NaN if unmatched |
| `noa_latitude` | Latitude of matched NOA hypocenter | Degrees |
| `noa_longitude` | Longitude of matched NOA hypocenter | Degrees |
| `noa_distance_km` | NOA `STA_distance` (degrees) × 111 converted to km | km |
| `mf_match_flag` | Whether an MF detection was matched within 2 s | `True`/`False`; ~4353 True across the full file |
| `mf_time_diff_s` | Time difference to nearest MF detection | Seconds; NaN if unmatched |
| `mf_template_name` | Template name of matched MF detection | `unmatched` if unmatched |
| `mf_real_mag` | MF regression magnitude (`Real_Mag`) of matched detection | NaN if unmatched |
| `mf_sccs` | Stacked cross-correlation value (`CCs_s_trig`) of matched MF detection | Dimensionless; negative values are normal |

---

### noa_catalog_labeled.csv (tab-separated, ~893 rows)

**Produced by**: Run_6 Script 2 (`2_label_NOA_with_template_and_detectByMForNot.py`).
**Consumed by**: Run_6 Scripts 3, 4.

Base source: NOA IMS file parsed by `extract_station_info(file, 'FSK')`, deduplicated by `EQ_time`, approximately 893 unique earthquake events after time-window filtering.

**NOA parsed columns**:

| Column | Meaning | Units/Notes |
|------|------|-----------|
| `EQ_time` | Earthquake origin time | UTC |
| `EQ_latitude` | Hypocenter latitude | Degrees, WGS84 |
| `EQ_longitude` | Hypocenter longitude | Degrees, WGS84 |
| `EQ_depth` | Hypocenter depth | km |
| `EQ_magnitude` | NOA magnitude (ML) | Dimensionless |
| `STA_name` | Station name (fixed as `FSK`) | — |
| `STA_distance` | Epicentral distance (**note: unit is degrees**; Script 4 multiplies by 111 to convert to km) | Degrees |
| `STA_azimuth` | Azimuth | Degrees |
| `STA_phase` | Phase type (e.g. `P_1`, `S_2`) | — |
| `STA_arrival_time` | Observed arrival time at FSK station | UTC |

**Columns appended by Script 2**:

| Column | Meaning | Units/Notes |
|------|------|-----------|
| `template_region` | MF template region name (e.g. `template_10_80Hz`) to which this event belongs; `outside_all_templates` if >8 km from all template centers | — |
| `dist_to_template_center_km` | Geodesic distance from hypocenter to nearest template center | km |
| `is_detected_by_mf` | Whether this NOA earthquake was detected by MF (time tolerance 3 s AND spatial ≤ 7 km) | `True`/`False`; ~251 True across the full file |
| `mf_detection_time_diff_s` | Time difference between MF detection and NOA `STA_arrival_time` | Seconds; NaN if unmatched |
| `mf_template_name` | MF template name for successful detections | `unmatched` if unmatched |

---

## IV. Figure File Descriptions

| File | Producing Script | Content |
|------|----------|------|
| `level_20/compare_figures/3_catalog_comparison_plot.png` | Script 3 | Multi-panel comprehensive comparison: one row per MF template (K1/K2/K3/I1/I2, 5 rows total) + one overview row; each row contains time–magnitude scatter, 2-hour event count histogram, FMD, and G-R cumulative frequency; MF=red, STA/LTA=green, NOA=blue. figsize=(55, 11×6), dpi=150 |
| `level_20/compare_figures/4_detection_power.png` | Script 4 (original) | Three-panel detection capability figure (vertical layout): full-width magnitude–distance plot at top + left-bottom inset + right-bottom map (extent `lat[36.5,40.5] lon[18.9,22.7]`; STA/LTA map markers in white stars). dpi=250 |
| `level_20/compare_figures/4_detection_power_flat_layout.png` | Script 4 (horizontal) | Three-panel detection capability figure (horizontal layout): large map on left full height (focused `lat[36.7,39.5] lon[19.95,22.2]`; STA/LTA in blue stars) + magnitude–distance plot top-right + inset bottom-right. dpi=250 |
| `level_20/compare_figures/4_detection_power_1111.png` | No corresponding script | No script outputs this filename; likely a manually saved debug version |
| `insert_fig.png` | Manually prepared | Inset figure embedded by Script 4 (original); not auto-generated by a script; must be included |
| `insert_fig_flat.png` | Manually prepared | Inset figure embedded by Script 4 (horizontal); not auto-generated by a script; must be included |

---

## V. Layout Differences Between the Two Versions of Script 4

| Difference | Original (`4_detection_power.png`) | Horizontal (`4_detection_power_flat_layout.png`) |
|--------|--------------------------------|---------------------------------------------|
| Figure size | 20×21 inches | 30×21 inches (wider) |
| Panel layout | Magnitude–distance plot spans top row; inset bottom-left; map bottom-right | Map occupies full left side; magnitude–distance plot top-right; inset bottom-right |
| Embedded inset | `insert_fig.png` | `insert_fig_flat.png` |
| Map extent | `lon[18.9,22.7] lat[36.5,40.5]` (larger) | `lon[19.95,22.2] lat[36.7,39.5]` (focused on Kefalonia) |
| STA/LTA map marker color | White stars | Blue stars |
| scaling_factor | 30 | 70 (larger symbols) |

Both versions are data-logically equivalent; use either one.
