# yuhan-mfdas (`yuhan_dasmf_v1.0`) — Top-Level Overview Document

> Kefalonia submarine DAS "matched filter (MF)" earthquake detection and cataloging pipeline — end-to-end overview of the public release v1.0.
> This version is a **single clean main-line** streamlined from the internal project `matched_filter_v8`: historical evolution branches and duplicate variants have been removed, and a **self-contained redesign** has been applied (see Chapter II), allowing the entire folder to run in isolation from external projects (depending only on external *data*).
> For details, refer to each subdirectory's `README.md`; this overview provides a bird's-eye view, instructions for running, and key design decisions.
> **GitHub version:** this repository contains the code and documentation only. The data and intermediate products described below are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip`, folder `yuhan_dasmf_v1.0/`); "included in this release" below refers to that archive. See `README.md` → *Data availability* for the files each step needs.

---

## 0. Quick Start and New Entry Points in This Version (Read First)

**One command to see a detection result (seconds, no large data / network needed):**

```bash
pip install -r requirements.txt     # see requirements.txt (core group is enough for demo)
python demo.py                       # run one template matching trigger using the 04-25 SCCs+MAD (from the Zenodo archive)
```

`demo.py` prints the strongest detection (2024-04-25 11:02:13 UTC, correlation +417×MAD = the template earthquake itself) and outputs `demo_detection.png`. This corresponds to Run_2 (triggering). In the GitHub version, first copy its two input files from the Zenodo archive (see `README.md`).

**New additions / key entry points in this version:**

| Path | What It Is |
| --- | --- |
| `demo.py` / `demo_detection.png` | One-command smoke test: see a real detection + result figure in seconds |
| `requirements.txt` | Python dependencies (core group for main pipeline; map group only for Run_6/paper figures) |
| `IN_DAS_data/` | DAS continuous raw data **sample** (2024-04-25 15:00–16:00, ~1.1 GB, for Run_1 demo), see its `README.md` (data in the Zenodo archive only; the GitHub folder holds only the notes) |
| `IN_NOA_cata/` | Input: NOA official earthquake catalog + channel mapping (not in the GitHub version: NOA bulletins are link-only, `kefa_chan_map.csv` is in the Zenodo archive) |
| `IN_STALTA_cata/` | Input: STA/LTA earthquake catalog `sta_lta_catalog.csv` (product of the STA/LTA method in companion repo `WANGYuhan2002/das_wavefront_picker` / Zenodo `stalta_multichan_v1/`, for comparison in Run_6/Run_4-3a/Run_2-2; in the Zenodo archive only) |
| `IN_noise_meteo/` | Input: noise product `noise_trends h5` + meteorological CSVs + `load_kefa_meteo.py` (from the spectral analysis package `WANGYuhan2002/yuhan_das_spectra`, for plotting in Run_4/5_* and Output_1/999; now self-contained and runnable). GitHub version: only `load_kefa_meteo.py` is here; the h5 and the two CSVs are in the Zenodo archive, and same-named copies are in the companion repository `WANGYuhan2002/yuhan_das_spectra` (`products/` and `meteo/`, folder `check_minmag_pkg/` inside `matched_filter_release.zip`) |
| `.dasmf_root` | Path self-adaptation sentinel (scripts use this to locate the project root; see Chapter II) |
| `LICENSE` | MIT open-source license (© 2026 Yuhan Wang) |

> The author-facing planning/migration notes of the Zenodo archive are not part of the GitHub version (they are not required for running the pipeline).

---

## I. One-Line Description and Overall Data Flow

**Description**: A small number of high-SNR, mutually dissimilar known earthquakes serve as "templates"; stacked cross-correlation (SCCs) sliding matching is performed along continuous strain-rate data from DAS fiber-optic cable; automatic detection, de-overlap, and magnitude estimation produce an earthquake catalog more complete than STA/LTA or the official NOA catalog, with quantitative assessment of detection capability.

**Core quantities used throughout:**

- **SCCs** (`CCs_s`): normalized cross-correlation between a template and continuous data at each time step, then averaged over channels; closer to ±1 = more similar to the template — the primary detection signal.
- **RAWCCs** (`RAWCCs_s` / `max_RAWCCs`): unnormalized version; amplitude is proportional to true ground motion amplitude, used for small-event magnitude estimation.
- **MAD**: noise baseline for SCCs; trigger threshold = `MAD × level` (`level_20` = 20×MAD).
- **max_stalta** (colloquially called "kurtosis" in this project; actually the peak STA/LTA ratio on SCCs): pulse sharpness, used to reject false positives.

**End-to-end pipeline (Run_X → Output_X):**

```
IN_NOA_cata/                      (Input: NOA official catalog IMS/EPI + DAS channel geographic mapping)
   │  100km IMS (template selection) / 200km IMS (magnitude calibration & comparison)
   ▼
Run_0_preparing_templates_from_catalog        1→2→3→4
   Inputs: 100km IMS + external STA/LTA catalog + external continuous DAS
   ▼
Output_0_Template_library                     1→2→3 (re-check / produce npy / add fields)
   Produces: ★Template_library.csv (27 columns) + ★Template_npy_files/*.npy (5 templates) + correlation matrix
   ▼
Run_1_compute_SCCs_and_MAD       1_..._io_once + 2_MAD_..._Parallel
   Inputs: template npy + external continuous DAS
   ▼
Output_1_SCCs_and_MAD
   Produces: ★SCCs_h5/<template>/CCs_stack_*.h5 (10-min windows, CCs_s/RAWCCs_s) + ★MAD_arrays/.../MAD_Average_dict.pkl
   ▼
Run_2_trigger_SCCs_kurtosis      1_..._Parallel
   Inputs: SCCs_h5 + MAD_Average_dict.pkl
   ▼
Output_2_Triggered_SCCs_points
   Produces: ★level_{N}/<template>/CCs_trig_*.pkl (trigger point dicts, containing max_stalta)
   ▼
Run_3_record_peakAmp             1_Record_Amp_Parallel
   Inputs: trigger points + template library + MAD + look up external continuous DAS
   ▼
Output_3_Trig_PeakAmp
   Produces: ★level_{N}/Trig_MAX_amp_df.csv (per-event peak amplitude) + per-event amp pkl
   ▼
Run_4_assign_overlaps_with_kurtosis           2a → 2b
   De-overlap + classify (2a) + estimate magnitudes (2b; + kurtosis to reject false positives,
   calibrate to NOA IMS 200km)
   Inputs: trigger point pkl + peak amplitude csv + template library + MAD/SCCs + 200km IMS
   ▼
Output_4_Classified_bySCCs_cleaned_overlaps_withKurtosis   ← MF final earthquake catalog
   Produces: ★level_20/Events_classified_AllTemp.csv + ★MagRegression/Events_classified_Real_Mag.csv
   │
   ├─► Run_5_fk_exclude_false_positive  ─► Output_5_CCs_fk   (f-k false-positive diagnostic figures, terminal)
   │
   └─► Run_6_get_and_compare_STALTAinMF_NOAinTemp_MF ─► Output_6_Compare_MF_stalta_NOA
          Inputs: Output_4 + external STA/LTA catalog + 200km IMS → three-catalog comparison / detection power figures (pipeline endpoint)

functions/ = shared function library for the entire pipeline (imported by each step, produces no data itself)
```

> **Numbering note**: This release has **renumbered the pipeline as consecutive Run_0 → Run_6 / Output_0 → Output_6** (the corresponding old numbers in the internal project `matched_filter_v8` were 0/1/2/3/7/8/9; numbers 4/5/6 belonged to historical variants that have been removed, causing gaps; this version eliminates the gaps for cleaner reading by external users).
>
> Regarding "de-overlap + classify + estimate magnitudes" (this release's **Run_4**): this step had four **parallel variants** in the internal project — basic version, +RAWCCs version, +kurtosis version, amplitude ablation version (internal old numbers: Run_4 / Run_6 / Run_7 / Run_7_amp respectively). **This release keeps only the final main-line "+kurtosis version", renumbered Run_4**; the other three have been removed. They were parallel alternative implementations, not a serial chain — Run_4 reads directly from Output_2/Output_3 with no dependencies on other variants.

---

## II. Self-Contained Redesign (Key Changes in v1.0 Relative to v8)

To allow this folder to run independently of `matched_filter_v8` and other legacy projects, the following changes were made (**only paths and import mechanisms were changed; no algorithm logic was altered**):

1. **Root sentinel** `.dasmf_root`: placed at the release root, used by scripts to locate the project root.
2. **"Path bootstrap" block injected at the top of every `.py`**: searches upward from `__file__` for `.dasmf_root` to find `PROJECT_ROOT`, then adds `PROJECT_ROOT` and `PROJECT_ROOT/functions` to `sys.path`. → Therefore **the entire folder can be moved to any location** and still run (self-adapting paths).
3. **All hardcoded `matched_filter_vN/` absolute paths** → replaced with `f"{PROJECT_ROOT}/..."` (114 instances total).
4. **`functions/` and `pydas_readers/` given `__init__.py`** to become proper packages; both `from readers import …` and `from functions.readers import …` styles work.
5. Dead code in `functions/append_cut_convert.py` — imports pointing to external v3 — have been updated to local (the file is unused project-wide).

Verification: all 60 `.py` files pass `py_compile`; core function modules import successfully under isolated paths; no remaining `matched_filter_vN` references. (A backup taken before the path fix, `_py_backup_before_pathfix.tar.gz`, was kept outside the release on the author's machine; it is not distributed.)

GitHub version: the remaining machine-specific data locations (raw DAS mount, MFFI data, external plotting repository) were also replaced by package-relative paths — see the table below; the same files are read and written.

### Still "External" Dependencies (Consistent with "Data May Be External" Convention; Note for Running/Reproducing)

| External Dependency | Path / Nature | Who Needs It | Notes |
|---|---|---|---|
| Continuous raw DAS data | 80 Hz archive `DATA_DEC_80Hz_16m_SORTED/`, read from `IN_DAS_data/continuous_scan_1h/` (8 Hz / 400 Hz products: `IN_DAS_data/DATA_DEC_8Hz_16m_SORTED/`, `IN_DAS_data/DATA_RAW_400Hz_8m/`) | Run_0/1/3/5, Output_0 | Primary input data, large volume; **not in release** (only the 1-hour sample is in the Zenodo archive); place or symlink the data at these package-relative paths |
| External STA/LTA catalog | copies of the external `stalta_clustering/.../sta_lta_catalog.csv`: `IN_STALTA_cata/sta_lta_catalog.csv` and `Run_0_.../threshold_100_trigNO_100_sta0.3s_lta3s/sta_lta_catalog.csv` | Run_0, Run_2 (plotting), Run_6 | Product of the STA/LTA detector (companion repo `WANGYuhan2002/das_wavefront_picker` / Zenodo `stalta_multichan_v1/`); the copies are in the Zenodo archive, not in this GitHub repository |
| NOA network connection | FDSN + contextily basemap | Run_6's `4_*` (via `functions/noa.py`) | Plotting requires online access for station coordinates / basemap tiles; will fail offline |
| External plotting repository | `GRL_DAS_MF_plots/functions/` (unpublished; fallback path `external/GRL_DAS_MF_plots/functions/`) | Output_0/`4_*_for_paper`, Run_4/`5_999_*` | Used only by paper-figure scripts, not in core pipeline. The three modules they import (`plot`, `load_das_h5_PE2024`, `compute_correlation`) also exist in `functions/`, which comes first on `sys.path`, so they are imported from there |
| External noise / meteorological data | `IN_noise_meteo/` (products of the `check_minmag_validity` project; in the Zenodo archive, and in companion repo `WANGYuhan2002/yuhan_das_spectra` / Zenodo `check_minmag_pkg/`), `IN_MFFI_data/kefalonia_ts_h5_80Hz/` (MFFI data from `mffi_kefa_analysis`, not distributed) | Run_4/`5_*`, Output_1/`999_*`, `das_mffi_io` | Used only for plotting / template construction assistance, not in core detection pipeline |

---

## III. Folder Roles + Script Selection Summary Table

> "Main-line scripts" = those that must be executed to run the pipeline; "Ignorable" = legacy/exploratory/plotting scripts, **kept in place as requested**, noted here and in each `README.md`, not deleted or moved.

| Folder | Role | Main-Line Scripts (to run) | Ignorable (kept in place) |
|---|---|---|---|
| **functions/** | Shared function library | All modules except the one below + pydas_readers/ | `append_cut_convert.py` (dead code) |
| **IN_NOA_cata/** | Input: NOA catalog + channel mapping | No scripts (data) | EPI_*.txt, kefa_chan_map.csv not read by main pipeline |
| **Run_0_…** | Template preparation | `1`→`2`→`3`→`4` | `3_1_…_for_paper` (plotting) |
| **Output_0_…** | Template library finalization + re-check | `1`→`2`→`3` | `4_…_for_paper` (plotting, requires the external plotting repo `GRL_DAS_MF_plots`) |
| **Run_1_…** | Compute SCCs + MAD | `1_…_Parallel_io_once`, `2_MAD_…_Parallel` | `1_…_Parallel` (v3 legacy), `2_MAD_…_calculation` (v3 legacy) |
| **Output_1_…** | SCCs+MAD data | No scripts (data) | `MAD_arrays/999_…` (plotting, requires external meteorological data) |
| **Run_2_trigger_SCCs_kurtosis/** | Triggering | `1_…_Parallel` | `1_…constThres` (v3 legacy non-parallel), `2_Show_…` (plotting, requires external STA/LTA) |
| **Output_2_…** | Trigger point data | No scripts (data) | — |
| **Run_3_…** | Record peak amplitudes | `1_Record_Amp_Parallel` | `1_Record_Amp` (v4 legacy) |
| **Output_3_…** | Peak amplitude data | No scripts (data) | Per-event amp pkl (not read downstream) |
| **Run_4_…_with_kurtosis/** | De-overlap + classify + estimate magnitudes | `2a_…_with_kurtosis`, `2b_Est_Mag_…` | `1`/`1a` (exploratory), `1b` (method rationale, recommended to keep), `2b_999`/`3a`/`3b`/`4_FMD`/`5_*` (plotting, incl. the slimmed `5_extract_Show_slip_momentum.py`) |
| **Output_4_…** | ★MF final earthquake catalog | No scripts (data) | `_addRAWCCs.csv` and `AllTemp.csv` are near-duplicates |
| **Run_5_…** | f-k false-positive rejection | `1_CCs_fk_…` + `spectrum.py` + `group_similarity.py` | `2_CCs_ZLCC_…` (exploratory), `3_…_for_paper` (plotting) |
| **Output_5_…** | False-positive diagnostic figures | No scripts (data/figures) | Bulk PNG files; may keep only contrast examples |
| **Run_6_…** | MF/STA-LTA/NOA comparison | `1`→`2`→`3`→`4_…_map` | `4_…_CHANGE_LAYOUT` (alternative layout, use either one) |
| **Output_6_…** | Comparison data/figures | No scripts (data/figures) | `insert_fig*` (manually prepared figures, must be included) |

---

## IV. How to Run + Sample Data Scope

**How to run**: go into any `Run_X/` directory and execute `python <main-line-script>.py` directly (scripts automatically locate `PROJECT_ROOT`, no environment variables needed). Execute in order Run_0 → Run_1 → Run_2 → Run_3 → Run_4 → (Run_5 / Run_6); prerequisite is having the external continuous DAS data at `IN_DAS_data/continuous_scan_1h/` (see Chapter II).

**Sample data scope (included in the Zenodo archive; none of these files are in the GitHub repository)**:

- Input: `IN_NOA_cata/` (complete; for the GitHub version obtain the NOA bulletins from NOA), `Output_0_Template_library/` (complete template library csv + 5 npy files).
- Intermediate data are mostly **example subsets**, recomputable from upstream scripts:
  - `Output_1/SCCs_h5/`: only **2024-04-25 (one full day)** included (5 templates × 144 10-minute h5 files ≈ 532 MB); full ~31 GB not included.
  - `Output_2`/`Output_3`: trigger points and amplitudes (by level).
  - `Output_4`: MF final earthquake catalog CSV (core result, recommended to keep complete).
  - `Output_5`/`Output_6`: diagnostic/comparison figure examples.

**Moving to another machine/path**: simply copy the entire folder (paths self-adapt); only the external data need to be provided at the package-relative locations of Chapter II (raw DAS under `IN_DAS_data/`, STA/LTA copies under `IN_STALTA_cata/`). (The pre-fix backup `_py_backup_before_pathfix.tar.gz` mentioned above is not distributed.)

---

## V. Input and Intermediate Data Inventory (in Pipeline Order)

| Step | Data Product | Format | Produced By | Consumed By |
|---|---|---|---|---|
| Input | `IMS_*.txt` (100km/200km) | NOA text | NOA website | Run_0 (100km), Run_4/Run_6 (200km) |
| Input | Continuous DAS h5 | h5 (80 Hz/16 m) | External archive, read from `IN_DAS_data/continuous_scan_1h/` (not in this repo) | Run_0/1/3/5 |
| Output_0 | ★`Template_library.csv` (27 columns) | CSV | Output_0 Script 3 | All downstream |
| Output_0 | ★`Template_npy_files/*.npy` | npy (time×channel) | Output_0 Script 2 | Run_1/3/5 |
| Output_1 | ★`SCCs_h5/<template>/CCs_stack_*.h5` | h5 (CCs_s/RAWCCs_s) | Run_1 | Run_2/3/5 |
| Output_1 | ★`MAD_arrays/.../MAD_Average_dict.pkl` | pkl | Run_1 | Run_2/3 |
| Output_2 | ★`level_{N}/<template>/CCs_trig_*.pkl` (with max_stalta) | pkl | Run_2 | Run_3/4/5 |
| Output_3 | ★`level_{N}/Trig_MAX_amp_df.csv` | CSV | Run_3 | Run_4's 2b |
| Output_4 | ★`Events_classified_AllTemp.csv` | CSV | Run_4's 2a | Run_4's 2b, Run_5/6 |
| Output_4 | ★`MagRegression/Events_classified_Real_Mag.csv` | CSV | Run_4's 2b | Run_5/6, plotting |
| Output_6 | `stalta_catalog_enriched.csv`, `noa_catalog_labeled.csv` | CSV | Run_6's 1/2 | Run_6's 3/4 |

**Naming shortcuts**: `level_{N}` = N×MAD threshold level; timestamp `UTC_YYYYMMDD_HHMMSS.ffffff` = 10-minute slice start; `Rt_trig` = true earthquake origin time after subtracting template S-wave offset; template display names `K1/K2/K3/I1/I2`.

---

## VI. Notes

1. **Terminology**: "kurtosis" throughout this project refers to the peak STA/LTA ratio `max_stalta` computed on SCCs, not the statistical fourth-moment kurtosis.
2. **Working level**: Output_2/Output_3 have multiple `level_*` directories from a parameter sweep; Run_4 defaults to writing `level_20`. If releasing only one level, keep `level_20`.
3. **Data redundancy**: `Events_classified_AllTemp.csv` and `_addRAWCCs.csv` are nearly identical in content (note: 2b currently reads the `_addRAWCCs` version).
4. **Runnability of plotting scripts externally**: any plotting script annotated as "requires the external plotting repository `GRL_DAS_MF_plots` / external noise-meteorological data / network connection" may not run outside the author's environment; these are not part of the core detection pipeline and are only needed for reproducing paper figures.
5. Each subdirectory's `README.md` has been updated to reflect this release's scope (annotated as main-line / ignorable, references to removed folders deleted).

---

*This overview and all `README.md` files are documentation only; the self-contained redesign changed only paths and import mechanisms, not any algorithm logic.*
