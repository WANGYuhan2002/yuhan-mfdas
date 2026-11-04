# IN_NOA_cata/ — Documentation

> **GitHub version:** the data files described here are not in this repository. The NOA reviewed bulletins (`100km_distance/`, `200km_distance/`, IMS1.0 format) are not redistributed (link only): obtain them from the National Observatory of Athens, Institute of Geodynamics (NOA-GEIN, https://www.gein.noa.gr/) and place them at the paths below. `kefa_chan_map.csv` is in the Zenodo archive [10.5281/zenodo.20715241](https://doi.org/10.5281/zenodo.20715241).

## I. Overview

This directory contains the **NOA (National Observatory of Athens) official reviewed earthquake catalog** for the Kefalonia area during the DAS deployment period (2024-04-25 to 2024-06-23). Two radius subsets are provided, each used by different stages of the pipeline.

```
IN_NOA_cata/
├── 100km_distance/
│   └── IMS_20240425_20240623_319.txt   # 728 events, ML 0.1–4.1
├── 200km_distance/
│   └── IMS_20240425_20240623_263.txt   # 2083 events, ML 0.1–4.3
└── kefa_chan_map.csv                    # 1705 channel lat/lon/elevation table (paper use only)
```

---

## II. Catalog Files

### 100km_distance / IMS_20240425_20240623_319.txt

- **Radius**: events within 100 km of the DAS cable.
- **Event count**: 728 events.
- **Magnitude range**: ML 0.1–4.1.
- **Station filter applied by parser**: FSK (Fiskardo, Kefalonia island).
- **Used by**: **Run_0** (template selection) — selecting NOA events with ML ≥ 2.0 + high SNR as template candidates; smaller radius is used because template earthquakes should be local to the DAS array.

### 200km_distance / IMS_20240425_20240623_263.txt

- **Radius**: events within 200 km of the DAS cable.
- **Event count**: 2083 events.
- **Magnitude range**: ML 0.1–4.3.
- **Used by**: **Run_4** (magnitude calibration via NOA ML matching) + **Run_6** (three-catalog comparison) — larger radius used because magnitude calibration and detection comparisons benefit from including all regional events.

---

## III. Catalog Format and Parser

Both files use the **IMS 1.0** format (also called `.ims` or bulletin format), the standard format used by NOA for distributing reviewed event bulletins.

**Parser**: `functions/extract_station_info.py`'s `extract_station_info(file_path, station_name)`:

```python
events = extract_station_info('IN_NOA_cata/100km_distance/IMS_20240425_20240623_319.txt', 'FSK')
# Returns a list of dicts, one per event-phase entry matching station 'FSK'
```

**Returned fields per event**:

| Field | Type | Units / Notes |
|---|---|---|
| `EQ_time` | `datetime` | Earthquake origin time (UTC) |
| `EQ_latitude` | float | Hypocenter latitude (degrees, WGS84) |
| `EQ_longitude` | float | Hypocenter longitude (degrees, WGS84) |
| `EQ_depth` | float | Hypocenter depth (km) |
| `EQ_magnitude` | float | NOA ML magnitude |
| `STA_name` | str | Station name (always `'FSK'` after filtering) |
| `STA_distance` | float | **Epicentral distance in DEGREES** (⚠ multiply by 111 to get km) |
| `STA_azimuth` | float | Station azimuth from epicenter (degrees) |
| `STA_phase` | str | Phase type (e.g. `'P_1'`, `'S_2'`) |
| `STA_arrival_time` | `datetime` | Observed phase arrival time at FSK (UTC) |

**Important**: `STA_distance` is in **degrees**, not kilometers. Code that uses this field must convert: `distance_km = STA_distance × 111`.

---

## IV. EPI Format Files (Archive Only)

Some older runs stored catalogs in a different `.epi` format (IMS extended event list). These are **not used by any main pipeline script** — they are historical archives only. The active pipeline uses only the IMS 1.0 `.txt` files above.

---

## V. `kefa_chan_map.csv` — Channel Mapping Table (Paper Use Only)

- **Contents**: 1705 rows, columns `Latitude`, `Longitude`, `Elevation`, `Name` (channel index). Maps each DAS channel index to its geographic coordinate.
- **Source**: Derived from fiber calibration data for the Kefalonia DAS cable.
- **Used by**: Paper-figure plotting scripts (map overlays, cable-trace visualization). **Not used by any main pipeline script** (Run_0 through Run_6). Channel selection in the pipeline is distance-based (channel index = distance / `dx`, e.g. `int(d1/dx):int(d2/dx)` in `functions/readers.py`) and does not use any mapping table. (The `Channel_mapping_information_catalouge_v4*.csv` table that `pydas_readers/mapping/channel_mapping.py` refers to belongs to a different DAS deployment, is not included, and `get_mapping()` is never called.)

---

## VI. Usage in the Pipeline

| Run step | Catalog file | Purpose |
|---|---|---|
| Run_0, Script 1 | `100km_distance/` | Match STA/LTA triggers against NOA to find ML ≥ 2.0 template candidates (SNR filter ≥ 30, peak amplitude ≥ 900) |
| Run_4, Script 2b | `200km_distance/` | Find NOA events within ±0.05° lat/lon + 5 s of MF detections for ML vs. Quasi_Mag regression (magnitude calibration) |
| Run_6, Scripts 1–4 | `200km_distance/` | Three-catalog comparison: label NOA events as detected/undetected by MF; co-plot NOA vs. MF vs. STA/LTA |

---

## VII. NOA Match Statistics (Run_4)

After Run_4 magnitude calibration with 200km catalog, per-template NOA match counts (≥3 matches required for regression):

| Template | NOA matches |
|---|---|
| template_0423_80Hz (K2, ML2.5) | 195 |
| template_10_80Hz (K1, ML2.4) | 121 |
| template_0425_80Hz (I2, ML3.0) | 38 |
| template_8_80Hz (K3, ML4.1) | 32 |
| template_20_80Hz (I1, ML3.1) | 4 |
