# Output_3_Trig_PeakAmp — Documentation

> **GitHub version:** the data files of this folder described below are not in this repository; they are in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241) (`matched_filter_release.zip` → `yuhan_dasmf_v1.0/Output_3_Trig_PeakAmp/`). Wherever this document says the data are bundled / included in the release, it refers to that archive. Copy the files back to the same relative paths to run the scripts that read them.

> Release path: `yuhan_dasmf_v1.0/Output_3_Trig_PeakAmp/`
> This is a pure data folder. All contents produced by scripts in `Run_3_record_peakAmp/`.

---

## I. Directory Role and Position in the Pipeline

This directory stores the **per-event peak DAS amplitude** results for all triggered events at each detection threshold level. These are used in the optional amplitude-based magnitude estimation branch (Quasi_Mag_Amp) of Run_4 — but are **not on the critical path for Real_Mag** (the final calibrated magnitude).

**Pipeline position**: Run_2 (triggering) → **Run_3 (this folder: peak amplitudes)** → Run_4 (magnitude estimation uses both this folder and the SCCs h5 files)

**Critical-path note**: `Real_Mag` (the final published magnitude) is computed from `max_RAWCCs × fac_CC2amp`, where `fac_CC2amp` comes from template constants in `Template_library.csv`, **not** from per-event DAS amplitudes in this folder. Run_3/Output_3 only feeds the optional `Quasi_Mag_Amp` branch in Run_4 Script 2b. Even though Run_4 Script 2b currently reads `Trig_MAX_amp_df.csv` unconditionally, the values it reads do not affect `Real_Mag` computation.

---

## II. Directory Structure

```
Output_3_Trig_PeakAmp/
├── level_15/
│   ├── config.json
│   ├── Trig_MAX_amp_df.csv             # ★ summary table (tab-separated)
│   ├── checkpoint.pkl                  # run resume state
│   └── template_0423_80Hz/
│       └── event_20240425T102312.pkl   # per-event amplitude dict
│       └── ...
│   ├── template_0425_80Hz/ ...
│   ├── template_10_80Hz/  ...
│   ├── template_20_80Hz/  ...
│   └── template_8_80Hz/   ...
├── level_20/   (same structure)
├── level_25/   (same structure)
├── level_30/   (same structure)
├── level_35/   (same structure)
├── level_40/   (same structure)   ← the level used in the main pipeline
└── level_50/   (same structure)
```

Seven threshold levels (15/20/25/30/35/40/50) are provided. `level_40` is the primary level used in Run_4 (consistent with the MAD×40 trigger threshold choice).

---

## III. Data Formats

### `config.json` — Run Parameters

Records the parameters used to produce this level's output:

```json
{
  "freq_band": [8, 20],
  "das_dist_start": 8300,
  "das_dist_end": 14000,
  "level": 40,
  "template_names": ["template_0423_80Hz", "template_0425_80Hz", "template_10_80Hz", "template_20_80Hz", "template_8_80Hz"]
}
```

Note: Run_3 uses `freq_band=[8,20]` (vs. the earlier `[5,20]` in older scripts) and all 5 templates (vs. 13 old templates).

---

### `Trig_MAX_amp_df.csv` — Summary Table (tab-separated)

One row per triggered event (all templates combined). 16 columns:

| Column | Description | Units / Notes |
|---|---|---|
| `CCs_s_trig` | SCCs value at trigger point | Dimensionless, ≈ ±0.01 |
| `RAWCCs_s_trig` | Raw cross-correlation value at trigger | Physical units |
| `t_trig` | Trigger time | UTC datetime string |
| `width_trig` | Trigger pulse width | timedelta |
| `t_left` | Left boundary of trigger window | UTC datetime |
| `t_right` | Right boundary of trigger window | UTC datetime |
| `max_RAWCCs` | Max `\|RAWCCs\|` (signed) within trigger window | Used indirectly |
| `t_arg_max` | Time of `max_RAWCCs` | UTC datetime |
| `max_stalta` | Max STA/LTA ratio on SCCs at trigger point | Dimensionless; used as quality criterion in Run_4 |
| `Rt_trig` | Trigger time as string key `YYYYMMDDTHHMMSS` | Used as join key with other tables |
| `Rt_left` | Left boundary as string key | |
| `Rt_right` | Right boundary as string key | |
| `CCs_s_trig_ofMAD` | `CCs_s_trig / MAD` | Normalized amplitude |
| `Template` | Template name | e.g. `template_10_80Hz` |
| `amp_max_mean` | Mean of per-channel peak amplitudes (349 channels, 8.3–14 km) | Strain-rate physical units |
| `amp_log_mean` | `mean(log10(\|amp_max\| + 1e−12))` per channel | Log-scale mean |

**Downstream usage** (Run_4 Script 2b reads this):
- `Rt_trig`, `Template`: join keys to merge with trigger catalog.
- `amp_log_mean`, `amp_max_mean`: used only for `Quasi_Mag_Amp` (optional branch).
- `max_RAWCCs`: also available here, but Run_4 primarily reads `max_RAWCCs` from the trigger pkl (Output_2), not from this CSV.

---

### Per-Event pkl Files — Peak Amplitude Dicts

Named `event_<YYYYMMDDTHHMMSS>.pkl`, one per event per template.

**Contents** (dict, 15+ keys). Key fields:

| Key | Type | Description |
|---|---|---|
| `amp_max` | `ndarray`, shape `(349,)`, float64 | Per-channel peak amplitude (absolute max in the bandpass-filtered window) for 349 channels in the 8.3–14 km range |
| `amp_max_mean` | float | `np.mean(amp_max)` — mean of per-channel peak amplitudes |
| `amp_log_mean` | float | `np.mean(np.log10(np.abs(amp_max) + 1e-12))` |
| Other fields | | DAS waveform metadata, channel distances, etc. |

The waveform extraction window is `[Rt_left − 1.5 s, Rt_right + 2.5 s]`; bandpass 8–20 Hz; channels at 8.3–14 km fiber distance (~349 channels at 80 Hz / ~16.34 m spacing).

---

## IV. Relationship with Magnitude Estimation

### Quasi_Mag_Amp (Optional Branch — Uses This Folder)

```python
# From Run_4 Script 2b
Quasi_Mag_Amp = amp_log_mean  # directly from Trig_MAX_amp_df.csv / per-event pkl
```

This is the "amplitude-based quasi-magnitude" — a simple log-scale proxy using per-event DAS amplitudes. It is stored in the output CSV but is not used as the final magnitude.

### Real_Mag (Critical Path — Does NOT Use This Folder)

```python
# From Run_4 Script 2b — uses template constants, not per-event amplitudes
fac_CC2amp = amp_max_mean / self_CC          # from Template_library.csv (constants)
Quasi_Mag  = log10(|max_RAWCCs × fac_CC2amp|)   # from Output_2 trigger pkl
Real_Mag   = slope × Quasi_Mag + intercept  # from NOA ML regression
```

`Real_Mag` requires only `max_RAWCCs` (from Output_2 trigger pkl) and template constants (from Output_0 Template_library.csv). Output_3 is bypassed entirely in this path.
