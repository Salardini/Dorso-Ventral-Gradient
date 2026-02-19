# A Dorsoventral Gradient of Rotational Dynamics in Human Cortex

Code and data for Salardini et al. *"A Dorsoventral Gradient of Rotational Dynamics in Human Cortex"*.

## Overview

This repository contains all analysis code, derived data, and figures for both the main manuscript and revision supplement. The cortical rotational dynamics index (rho) — computed from delay-embedded VAR(1) models of MEG resting-state source-reconstructed timeseries — exhibits a strong dorsoventral gradient (rho_s = -0.73, p < 10^-60) across Schaefer 400 parcels. The revision identifies the dual-origin structural covariance gradient (G2; Valk et al., 2020) as the key architectural predictor of this gradient.

## Repository Structure

```
dorsoventral_gradient/
├── meg_axes/              Core library: VAR model, tau/rho metrics, preprocessing, source reconstruction
├── atlas/                 Schaefer-400 parcellation utilities and fsaverage centroids
├── config/                Pipeline YAML configurations
│
├── scripts/
│   ├── pipeline/          MEG processing pipeline (stages 00-05 + batch runners)
│   ├── analysis/          Main manuscript post-hoc analyses (01-14 + utils)
│   ├── standalone/        Standalone analyses: fMRI replication, adaptive delay, atlas sensitivity,
│   │                      computational model, AHBA gene expression, figure generation
│   └── revision/          Revision supplement: dual-origin connectivity architecture (01-11)
│
├── data/
│   ├── main/              Main manuscript derived data (group maps, correlations, replication results)
│   └── revision/          Revision data (rho_master.csv, Valk gradients, BigBrain, parcellation labels)
│
├── figures/
│   ├── main/              Main manuscript figures (Figures 1-2, S1-S6, analysis panels)
│   └── revision/          Revision figures (dual-origin, SC, spin test panels)
│
├── writing/               Revision text drafts (methods, discussion, analysis summaries)
├── MANIFEST_main.md       Detailed file-by-file descriptions of main manuscript components
├── requirements.txt       Python dependencies
└── setup.py               Package installation
```

## Main Manuscript Pipeline

### MEG Processing (scripts/pipeline/)

```
00_extract_tarballs.py    Extract BIDS archives
01_reconall.sh            FreeSurfer cortical surface reconstruction
02_make_bem.py            Single-shell BEM forward model
    (manual coregistration step)
04_extract_parcels_and_metrics.py   Source reconstruction (dSPM) + Schaefer-400
                                    parcellation + tau (ACF integral) and rho
                                    (delay-embedded VAR(1) rotational index)
05_group_stats.py         Group means, spatial correlations, spin permutation tests
run_batch.py              Parallel batch processing with dependency management
```

### Core Library (meg_axes/)

- **metrics.py** — Tau: ACF integral (5-300 ms). Rho: delay-embedded VAR(1) eigenvalue analysis (rotational index). QC metrics.
- **preprocessing.py** — BIDS loading, notch filter (50/60 Hz), bandpass (1-40 Hz), resampling (200 Hz), optional ICA.
- **source.py** — Forward model (BEM), dSPM inverse solution, parcel time series extraction with PCA-flip.
- **config.py** — Nested dataclass configuration system with YAML + CLI override support.

### Post-Hoc Analyses (scripts/analysis/ 01-14)

01. Spatial correlations with spin tests (rho vs DV: r = -0.72)
02. Spectral confound controls (1/f slope, band power)
03. Task replication (auditory, visual conditions)
04. Principal gradient comparison
05. Figure generation
06. Nonlinear validation
07. Gradient axis angles
08. Regional tau-rho relationship
09. Critical spectral test
10. Split-half reliability
11. Internal consistency
12. Depth bias control
13. Axis angle validation
14. Spectral feature computation

### Standalone Analyses (scripts/standalone/)

- **fMRI replication** — MOUS fMRI (fmri_replication_pipeline_v2.py) and HCP ~1000 subjects (hcp_rho_replication.py)
- **Atlas sensitivity** — Schaefer 100/200/400 resolution tests (atlas_sensitivity_analysis.py)
- **Adaptive delay** — Tests whether band-specific gradients are embedding artifacts (adaptive_delay_analysis.py)
- **Computational model** — E-I network demonstrating tau-rho tradeoff (computational_model_tau_rho.py)
- **Gene expression** — AHBA PV/SST interneuron markers vs rho (ahba_gene_expression_analysis.py)
- **HCP MEG** — Tau/rho from HCP CIFTI data, Yeo17 parcellation (06_hcp_tau_rho.py)

## Revision Supplement: Dual-Origin Connectivity Architecture (scripts/revision/)

01. Gene expression (AHBA PVALB/SST)
02. FC metrics (gradients, graph metrics, Mesulam)
03. Valk structural covariance gradients (primary G2 analysis)
04. BigBrain MPC
05. Comprehensive multi-predictor figure
06. Spectral/tau progressive partial correlations
07. Spin permutation tests (5,000 permutations)
08. Structural connectivity (HCP SC via ENIGMA Toolbox)
09. BigBrain with proper Glasser mapping
10. Publication figure
11. Brain surface plots

Key finding: The dual-origin G2 is the only predictor surviving controls for spatial position, spectral exponent, and intrinsic timescale (r|z+SE+tau = -0.249, p_spin = 0.018).

## Data Sources

| Resource | Source | Used in |
|---|---|---|
| MEG MOUS dataset (ds004998) | [OpenNeuro](https://openneuro.org/datasets/ds004998) | Main pipeline |
| HCP MEG resting-state | [Human Connectome Project](https://www.humanconnectome.org/) | HCP replication |
| HCP fMRI (Schaefer-400 timeseries) | [Tipnis et al. 2022](https://doi.org/10.6075/J0C24WMW) | fMRI replication |
| Schaefer 400 parcellation | [CBIG](https://github.com/ThomasYeoLab/CBIG) | All analyses |
| Valk et al. (2020) structural covariance | [GitHub](https://github.com/sofievalk/projects) | Revision script 03 |
| HCP structural connectivity | [ENIGMA Toolbox](https://github.com/MICA-MNI/ENIGMA) | Revision script 08 |
| Allen Human Brain Atlas | [abagen toolbox](https://github.com/rmarkello/abagen) | Scripts 01 (main + revision) |
| BigBrain intensity profiles / MPC | [MICA-MNI/micaopen](https://github.com/MICA-MNI/micaopen) | Revision scripts 04, 09 |
| Mesulam classification | BrainSpace `mesulam_conte69.csv` | Revision script 02 |

## Requirements

```
# Core
numpy scipy pandas scikit-learn matplotlib seaborn

# MEG processing
mne mne-bids nibabel

# Spatial statistics / neuroimaging
brainspace brainsmash neuromaps

# Revision-specific
abagen enigmatoolbox

# Configuration
pyyaml
```

Full pinned versions in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
pip install -e .  # Install meg_axes package
```

## Notes

- Individual subject MEG timeseries (parcel_ts.npy, ~204 subjects) and per-subject metric CSVs are not included due to size. They can be regenerated by running the pipeline on the MOUS dataset.
- See `MANIFEST_main.md` for detailed per-file descriptions of all main manuscript components.
- See `writing/revision_README.md` for the original revision supplement README with full results table.
