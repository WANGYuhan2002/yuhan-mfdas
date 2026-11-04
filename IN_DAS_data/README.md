# IN_DAS_data — Continuous DAS Raw-Data Sample

> **GitHub version:** raw DAS data are never committed to the GitHub repository, so here this folder holds only this note and `说明文档.md`. The sample described below is part of the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/IN_DAS_data/continuous_scan_1h/`); copy it into this folder to run the Run_1 demo. Below, "bundled" / "included" refers to that archive.

This folder holds a **sample of the continuous DAS raw data** needed to run the pipeline
(80 Hz, channel spacing dx ≈ 16.34 m, 936 channels, strain rate). The full dataset
(2024-04-23 → 06-22, ~59 days, ~1.4 TB) is **not** bundled with the release; only a short,
demonstrable segment is included here.

## Directory layout

```
IN_DAS_data/
└── continuous_scan_1h/          1-hour continuous sample (for the Run_1 matched-filter scan demo)
    └── 20240425/                2024-04-25 15:00–16:00 (the most seismically active hour that day, 205 MF events)
        └── Gr_80Hz_16m_20240425_HHMMSS.861.h5   × 13 files (~84 MB each / 5 min, ~1.1 GB total)
```

> Files follow the loader's `<input_dir>/YYYYMMDD/*.h5` convention; the trailing
> `..._YYYYMMDD_HHMMSS.fff.h5` in each filename is parsed by the loader as that file's start
> time (one file per 5 minutes). The 13 files here cover 14:59:55–16:04:55, fully containing
> 15:00:00–16:00:00.

## Why this particular hour

- **04-25**: aligns with the intermediate products already bundled — `Output_1/SCCs_h5/` ships
  exactly one full day (2024-04-25) of stacked cross-correlation series, so the raw DAS for this
  day lets you "recompute SCCs from the big data" and cross-check file-by-file against the
  existing `SCCs_h5/*/...20240425...h5`.
- **15:00–16:00**: the hour with the most earthquakes detected on 04-25 (205 MF events) — a good
  window for a detection demo.

## How to use it (data flow is fully visible — no config, no environment variables)

- **Run_1** (`1_Compute_SCCs_10min_h5_Parallel_io_once.py`) does the matched-filter scan. It reads
  raw DAS through `functions/readers.py` → `Read_DAS_data`, which reads this folder
  `IN_DAS_data/continuous_scan_1h/` by default (in the Zenodo v1.0.0 code it pointed at the author's
  archive mount `.../DATA_DEC_80Hz_16m_SORTED/`). Limit the scan window (`t_all_start`/`t_all_end`
  in the script's `Config`) to 2024-04-25 15:00–16:00.
- The remaining detection steps (Run_2 trigger, Run_3 amplitude, Run_4 classify/magnitude,
  Run_6 comparison) do **not** need continuous DAS — they consume the intermediate products
  already bundled (`Output_1/2/3/4`). (Correction, GitHub version: re-running Run_3 itself does read
  raw DAS at each trigger time via `Read_DAS_data`; Run_4 onwards can start from the bundled `Output_3/`.)

## What is not included

- **Template-event raw files** (one ~84 MB raw h5 per template, 5 templates spanning
  2024-04-25/05-10/05-11/05-30, ~423 MB total) are **not yet bundled** — the earlier
  "template selection" step (Run_0) also uses a wider range of DAS, whose extent is still under
  the author's consideration. Until then, the release already ships the finalised templates
  (`Output_0_Template_library/Template_npy_files/*.npy`), so you can start from Run_1 without
  building templates yourself.
- 8 Hz / 400 Hz_raw and other data products are not used by this pipeline and are not included.

## Data source / citation

Kefalonia seafloor DAS, April–June 2024, Silixa iDAS v2. See the paper (preprint):
*Persistent Faulting Activity Along the Kefalonia Transform Fault Revealed by Seafloor DAS*
(Wang et al., 2026, https://doi.org/10.22541/essoar.15006791/v1).
