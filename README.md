# Automated Earthquake Catalog — Salton Sea Geothermal Region
**AI-PAL: Deep Learning Phase Picking & Association**

This repository contains the full AI-PAL (Automated Intelligent Phase-picking and Association Locator) pipeline for building an automated earthquake catalog in the Salton Sea Geothermal Region. The pipeline uses a SAR (Self-Attention RNN) deep learning model to detect P and S wave arrivals from continuous seismic data, followed by PAL association to cluster picks into earthquake events.

The demo runs on 8 days of continuous seismic data (August 23–30, 2012), capturing the **2012 Brawley Seismic Swarm** — one of the most significant seismic sequences in Southern California that year.

---

## Results

| Metric | Value |
|--------|-------|
| Total picks | 653,782 (full 2012) |
| Events detected | 9,416 (min_sta=3) |
| Peak day | Aug 27, 2012 — 11,370 picks |
| P-wave F1 | 95.89% |
| S-wave F1 | 95.14% |
| Both P+S detected | 95.98% of events |

---

## Pipeline

```
Continuous MiniSEED Waveforms (CI, XD, ZY networks)
    ↓
SAR Phase Picker (BiGRU + Multihead Self-Attention)
    ↓ 
P & S arrival picks (.pick files)
    ↓
PAL Associator (origin time clustering + grid search)
    ↓
Earthquake Catalog (catalog.csv)
    ↓
Visualization (maps, Gutenberg-Richter, swarm timeline)
```

---

## Model

**Architecture:** SAR (Self-Attention RNN) — Zhou et al. (2025), JGR Solid Earth  
**DOI:** [10.1029/2025JB031294](https://doi.org/10.1029/2025JB031294)

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer BiGRU + 4-head Multihead Self-Attention |
| Window length | 30s at 100 Hz |
| Frequency band | 2–45 Hz |
| RNN steps | 296 |
| Training data | CEED dataset, SCEDC 2008–2012 (56,554 samples) |
| Best checkpoint | Epoch 17/20, Loss = 0.0201 |

---

## Data & Model (Pelican OSDF)

All data and the model checkpoint are publicly accessible via Pelican OSDF — **no download required**:

```
Data  : osdf:///ndp/public/ucr_seis/Data_Salton
Model : osdf:///ndp/public/ucr_seis/models/8700_17-319.ckpt
```

| Date | Type | Expected Picks |
|------|------|---------------|
| Aug 23–25 | Quiet baseline | ~500–1,000/day |
| Aug 26 | Swarm onset | ~9,000 |
| Aug 27 | **Swarm peak** | ~11,370 |
| Aug 28–30 | Swarm decay | ~3,000–6,000/day |

---

## Repository Structure

```
├── models.py              # SAR model architecture
├── picker.py              # SAR phase picker
├── associator.py          # PAL associator
├── salton_sea_demo.ipynb  # Full demo notebook (15 cells)
└── config/
    └── salton_sea.sta     # 29-station file (CI, XD, ZY networks)
```

---

## Running on NDP

### 1. Start a JupyterHub server
- Go to [NDP JupyterHub](https://jupyterhub.nrp-nautilus.io)
- Use the custom image: `bpari004/seisai:v1.1`
- Recommended: 1 GPU, 8 cores, 32 GB RAM, amd64, /dev/shm checked

### 2. Clone this repository
```bash
cd /home/jovyan/work/_User-Persistent-Storage_CephBlock_/
git clone https://github.com/bpari004/UCOP_Automated-Earthquake-Catalog-Salton-Sea-Geothermal-Region.git
cd UCOP_Automated-Earthquake-Catalog-Salton-Sea-Geothermal-Region
mkdir -p config
mv salton_sea.sta config/
```

### 3. Open and run the notebook
Open `salton_sea_demo.ipynb` in JupyterLab and run cells sequentially.

- **Cell 3** is the only cell that may need editing (paths are pre-configured for NDP)
- **Data and model stream directly from Pelican OSDF** — no manual download needed
- Skip Cell 4 if running locally with data already downloaded

---

## Running Locally

```bash
conda activate your_env
cd /path/to/repo
jupyter lab
```

Update paths in **Cell 3** to point to your local data directory.

---

## Requirements

```
obspy>=1.4.0
torch>=2.0
numpy>=1.24,<2.0
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
cartopy
tqdm
fsspec
pelicanfs
```

Install via:
```bash
pip install obspy torch numpy scipy pandas matplotlib cartopy tqdm fsspec pelicanfs
```

---

## Station Network

- **Total stations:** 29
- **CI network:** 11 stations (HH? channels)
- **XD network:** 17 stations (BH? channels)  
- **ZY network:** 1 station — COON (EH? channels)

---

## Citation

If you use this pipeline, please cite:

> Zhou, Y., et al. (2025). AI-PAL: Automated Intelligent Phase-picking and Association Locator. *JGR Solid Earth*. doi:10.1029/2025JB031294

---

## Contact

**Binayak Parida**  
University of California, Riverside  
Department of Earth and Planetary Sciences
