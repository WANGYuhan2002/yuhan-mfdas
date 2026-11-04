# Output_5_CCs_fk — Documentation

> **GitHub version:** the data and figure files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_5_CCs_fk/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> This folder is the output directory of `Run_5_fk_exclude_false_positive/`. It stores visualization results for "false-positive diagnosis" applied to template matching trigger points. All contents are PNG images and config.json files — **no array or catalog intermediate data is stored here, and no downstream script consumes this directory** (terminal product).

---

## I. Folder Overview and Position in the Overall Pipeline

### Upstream (Input Sources)

This directory's contents are produced by scripts in `Run_5_fk_exclude_false_positive/`. All three scripts share the same upstream inputs:

| Input Data | Source Path | Purpose |
| --- | --- | --- |
| Template library table | `Output_0_Template_library/Template_library.csv` | `infile_name→plot_name` mapping, `t_P_sec/t_S_sec`, `plot_order` |
| Template waveforms | `Output_0_Template_library/Template_npy_files/template_*.npy` | Read via `Read_template` and bandpass-filtered; used to recompute `CCs_f` |
| Stacked cross-correlation SCCs | `Output_1_SCCs_and_MAD/SCCs_h5/` (10-minute slice h5 files) | Plot long/short time-series of SCCs; compute STA/LTA |
| Average MAD dict | `Output_1_SCCs_and_MAD/MAD_arrays/.../MAD_Average_dict.pkl` | Convert trigger threshold to `Threshold_level × MAD`; compute `CCs_s_trig_ofMAD` |
| Trigger points (core) | `Output_2_Triggered_SCCs_points/level_20/<template>/CCs_trig_*.pkl` | Per-trigger `t_trig/t_left/t_right/CCs_s_trig/max_stalta` etc. |
| Continuous raw DAS data | 80 Hz archive `DATA_DEC_80Hz_16m_SORTED/`, default location `IN_DAS_data/continuous_scan_1h/` (external data, time-addressed by `Read_DAS_data`) | Recompute per-channel cross-correlation matrix `CCs_f` at trigger times; plot waterfall |

### Downstream (Output Destination)

**No other script reads this directory.** This directory is the terminal diagnostic/display branch of the pipeline; its products are for manual inspection or publication figures, and do not enter subsequent cataloging steps.

---

## II. Subdirectory Structure and Producing Scripts

This directory has two subdirectories with actual output (`2_CCs_ZLCC/` does not exist — the corresponding script produced no results):

### `1_CCs_fk_spec/`

Produced by `Run_5_fk_exclude_false_positive/1_CCs_fk_for_trig_points.py` **[main-line script]**.

**Core discriminant**: 2D f-s (frequency–slowness) amplitude spectrum. For trigger points falling within specified correlation/STA-LTA ranges, the per-channel cross-correlation matrix `CCs_f` is recomputed, the amplitude spectrum is computed via `spectrum.Calculate_fk_amplitude_spectrum`, and multi-panel diagnostic figures are saved (waveform waterfall + SCCs + STA/LTA + CCs matrix + f-s spectrum).

**Directory contents**:

- `config.json`: records all parameters and paths for this run (note: the `dir_byPara` field recorded in config.json may differ from the actual parameter subdirectory name on disk; use the disk subdirectory name as authoritative).
- `kur1_12_C0.1_1_win0.5_0.5/`: 17 PNG files (I1×3, I2×4, K1×5, K2×5) — the actual output.

### `3_CCs_kurtosis_example_for_paper/`

Produced by `Run_5_fk_exclude_false_positive/3_CCs_kurtosis_example_for_paper.py` **[plotting script]**.

**Core discriminant**: Kurtosis (STA/LTA) peak. Publication-quality version with refined fonts/layout/relative time axis; the right panel shows a zoomed STA/LTA curve and the local peak (green star) within the trigger window.

**Directory contents**:

- `config.json`: records run parameters (`kur20_1000_C0.03_1_win1.5_1.5`).
- `kur20_1000_C0.03_1_win1.5_1.5/`: 25 PNG files (5 per template × 5 templates) — high-kurtosis examples; the formal output under the current parameters.
- `kur1_10_C0.03_0.1_win1.5_1.5/`: 25 PNG files (5 per template × 5 templates) — low-correlation/low-kurtosis examples; a second batch under different parameters, forming a true/false-positive contrast pair with the high-kurtosis subdirectory.

---

## III. Parameter Subdirectory Naming Convention

```text
kur{kur0}_{kur1}_C{C0}_{C1}_win{winL}_{winR}/
Example: kur20_1000_C0.03_1_win1.5_1.5/
```

| Prefix | Meaning | Source Parameter |
| --- | --- | --- |
| `kur<a>_<b>` | STA/LTA ("kurtosis") selection range [a, b] | `kurtosis_range` |
| `C<lo>_<hi>` | Stacked correlation `\|CCs_s_trig\|` selection range [lo, hi] | `C_trig_range` |
| `win<L>_<R>` | Data read window length left/right of the trigger time (seconds) | `Search_win` |

(If produced by Script 2, the subdirectory name also appends `_Z{Zl}_{Zr}_p{p}_q{q}` for group-similarity sliding-window and quantile parameters.)

---

## IV. Output PNG Naming Convention

```text
<plot_name>_<|CCs_s_trig|>_<CCs_s_trig_ofMAD>_<YYYYMMDD_HHMMSS_ffffff>.png
Example: Temp_I1_0.1136_50.3_20240511_135123_025000.png
```

Field meanings (verified against script code and `Template_library.csv`):

| Field | Example | Meaning and Units |
| --- | --- | --- |
| `plot_name` | `Temp_I1` | Template display name from the `plot_name` column of `Template_library.csv`; **not a slowness or azimuth** |
| `\|CCs_s_trig\|` | `0.1136` | Absolute value of the stacked cross-correlation at this trigger point; 4 decimal places; dimensionless (correlation coefficient) |
| `CCs_s_trig_ofMAD` | `50.3`, `−51.5` | Correlation value divided by the template's MAD (signed; negative = anti-correlated); 1 decimal place; dimensionless |
| Timestamp | `20240511_135123_025000` | Trigger time `t_trig`, precise to microsecond (`%Y%m%d_%H%M%S_%f`) |

**Template `plot_name` mapping** (from `Template_library.csv`):

| plot_name | infile_name | Corresponding event |
| --- | --- | --- |
| Temp_K1 | template_10_80Hz | 2024-05-30 Mw2.4 |
| Temp_K2 | template_0423_80Hz | 2024-04-25 Mw2.5 |
| Temp_K3 | template_8_80Hz | 2024-05-10 Mw4.1 |
| Temp_I1 | template_20_80Hz | 2024-05-11 Mw3.1 |
| Temp_I2 | template_0425_80Hz | 2024-04-25 Mw3.0 |

---

## V. Terminology Clarification

The directory prefix `kur*`, the script parameter `kurtosis_range`, and "Kurtosis" in figure titles all refer to the **unnormalized centered STA/LTA** (computed by `sta_lta_centred_unnorm`, sta=0.5s, lta=1.5s) — a measure of waveform impulsiveness/sharpness. **This is NOT the fourth-moment statistical kurtosis.**

---

## VI. Inventory of Existing On-Disk Products

```text
Output_5_CCs_fk/
├── 1_CCs_fk_spec/
│   ├── config.json                          (parameter record; may differ from actual subdirectory name)
│   └── kur1_12_C0.1_1_win0.5_0.5/          (17 PNG files: I1×3, I2×4, K1×5, K2×5)
└── 3_CCs_kurtosis_example_for_paper/
    ├── config.json                          (records kur20_1000_C0.03_1_win1.5_1.5)
    ├── kur1_10_C0.03_0.1_win1.5_1.5/       (25 PNG files, low-correlation/low-kurtosis contrast examples)
    └── kur20_1000_C0.03_1_win1.5_1.5/      (25 PNG files, high-kurtosis formal examples)
```

(`2_CCs_ZLCC/` does not exist on disk; the corresponding exploratory script produced no results.)
