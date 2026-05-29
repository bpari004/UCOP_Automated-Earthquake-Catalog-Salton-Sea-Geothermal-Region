# UCOP Automated Earthquake Catalog — Salton Sea Geothermal Region

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-bpari004%2Fseisai%3Av1.0-blue)](https://hub.docker.com/r/bpari004/seisai)
[![DOI](https://img.shields.io/badge/DOI-10.1029%2F2025JB031294-green)](https://doi.org/10.1029/2025JB031294)

---

## Overview

This repository contains a fully reproducible deep learning earthquake detection pipeline applied to the **August 2012 Brawley Seismic Swarm**, Salton Sea Geothermal Region, Southern California.

The pipeline uses **AI-PAL** (Zhou et al., 2025) — a self-attention RNN (SAR) phase picker combined with a Pick-and-Associate-Locate (PAL) associator — to automatically detect and locate earthquakes from continuous seismic waveforms.

The entire workflow runs on the **National Data Platform (NDP) JupyterHub** with seismic data and model checkpoint streamed directly from **Pelican OSDF**.

---

## Results

| Metric | Value |
|--------|-------|
| Period covered | Aug 23–30, 2012 (8 days) |
| Stations | 30 (CI, XD, ZY networks) |
| Total picks | 31,009 |
| Total events | 4,277 |
| Peak day | Aug 27 — 8,321 picks (Brawley Swarm peak) |
| Magnitude range | M−1.0 – M5.4 |
| Depth range | 2 – 24 km |
| Wall time (picking) | ~34 min on RTX 2080 Ti |
| Wall time (association) | ~3 min |

---

## Pipeline

```
Pelican OSDF (MiniSEED)
        ↓
SAR Phase Picking          picker.py + models.py
        ↓
picks/YYYY-MM-DD.pick      31,009 P+S picks
        ↓
PAL Association            associator.py
        ↓
output/catalog.csv         4,277 located events
```

---

## Data & Model

All data and model weights are publicly accessible via Pelican OSDF — no download needed:

| Resource | Pelican Path |
|----------|-------------|
| Seismic data (8 days MiniSEED) | `osdf:///ndp/public/ucr_seis/Data_Salton` |
| SAR model checkpoint | `osdf:///ndp/public/ucr_seis/models/8700_17-319.ckpt` |

**Station networks:**
- **CI** — Southern California Seismic Network (11 stations, HH?)
- **XD** — Dense geothermal deployment (18 stations, BH?)
- **ZY** — COON single station (1 station, EH?)

---

## SAR Model

| Parameter | Value |
|-----------|-------|
| Architecture | BiGRU (2 layers, hidden=128) + Multihead Self-Attention (4 heads) |
| Window length | 30s at 100 Hz |
| Frequency band | 2–45 Hz |
| RNN steps | 296 |
| P-wave F1 | 95.89% |
| S-wave F1 | 95.14% |
| Checkpoint | `8700_17-319.ckpt` (Epoch 17/20, loss=0.0201) |

Model trained on CEED (California Earthquake Event Dataset, 2008–2012, Salton Sea region).

---

## Running on NDP JupyterHub

### 1. Start a server

In NDP JupyterHub, launch a server with:

| Setting | Value |
|---------|-------|
| GPU | 1 × NVIDIA A100 80GB (or any available) |
| Cores | 8 |
| RAM | 32 GB |
| /dev/shm | ✅ checked |
| Custom image | `bpari004/seisai:v1.0` |

### 2. Upload files to persistent storage

Copy all files to:
```
/home/jovyan/work/_User-Persistent-Storage_CephBlock_/salton_sea_demo/
```

Place `salton_sea.sta` and `CFM7.0_traces.lonLat` in the `config/` subfolder.

### 3. Run the notebook

Open `salton_sea_demo.ipynb` and run cells in order. All data and the model checkpoint are streamed directly from Pelican — no manual downloads needed.

---

## Repository Structure

```
├── salton_sea_demo.ipynb       # Main demo notebook (15 cells)
├── models.py                   # SAR model architecture
├── picker.py                   # SAR phase picker (Pelican-aware)
├── associator.py               # PAL associator
├── config/
│   ├── salton_sea.sta          # 30-station file (lat, lon, ele, gain)
│   └── CFM7.0_traces.lonLat    # Community Fault Model v7.0 traces
├── picks/
│   └── YYYY-MM-DD.pick         # Per-day pick files (31,009 total)
└── output/
    ├── catalog.csv             # PAL catalog (ot, lat, lon, dep, mag)
    ├── catalog_relocated.csv   # Relocated catalog (4,277 events)
    ├── phase.dat               # Phase file for relocation
    ├── station_map.png         # Station network map
    ├── waveform_comparison.png # Quiet vs swarm day waveforms
    ├── daily_picks.png         # Daily pick count bar chart
    ├── epicenter_map.png       # Epicenter map + magnitude-time plot
    └── gutenberg_richter.png   # G-R relation + b-value
```

---

## Output Figures

### Station Network Map
30 stations across CI, XD, ZY networks with CFM7.0 fault traces and satellite basemap.

### Daily Pick Counts
Clear seismic swarm signature — picks jump from ~1,700/day (quiet) to 8,321 on Aug 27 (swarm peak), a **5× increase**.

### Epicenter Map
4,277 events tightly clustered in the Brawley Seismic Zone at the junction of the San Andreas and Imperial fault systems.

### Gutenberg-Richter Relation
b-value fit above magnitude of completeness (Mc ≈ 0.5), consistent with regional seismicity.

---

## Requirements

All packages are pre-installed in the Docker image `bpari004/seisai:v1.0`:

```
obspy>=1.4.0        torch>=2.0         zarr>=2.14
numpy>=1.24         scipy>=1.10        pandas>=2.0
matplotlib>=3.7     cartopy            h5py>=3.8
fsspec              pelicanfs          tqdm
```

---

## Reference

Zhou, W., et al. (2025). AI-PAL: Automated seismic phase picking and association using self-attention RNN. *Journal of Geophysical Research: Solid Earth*. https://doi.org/10.1029/2025JB031294

---

## Author

**Binayak Parida**  
University of California, Riverside  
Department of Earth and Planetary Sciences
