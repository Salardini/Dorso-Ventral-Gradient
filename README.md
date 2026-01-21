# A Dorsoventral Gradient of Rotational Dynamics in Human Cortex

Code and data for reproducing analyses from:

**Salardini A., et al. "A dorsoventral gradient of rotational dynamics in human cortex."** *(in preparation)*

## Key Findings

1. **Rho-DV Gradient**: Rotational dynamics (rho) vary systematically along the dorsal-ventral axis, with ventral regions showing stronger rotational (oscillatory) activity patterns
2. **Band Specificity**: The gradient is strongest in high-frequency bands (beta_high: r = -0.77, gamma_low: r = -0.73)
3. **Task Generalization**: The gradient replicates across resting-state (N=208), visual task (N=69), and auditory task (N=95) conditions
4. **Beyond Spectral Confounds**: ~32% of the gradient persists after controlling for all spectral features

## Repository Structure

```
Dorso-Ventral-Gradient/
├── data/                          # Processed group-level data
│   ├── parcel_group_maps.csv      # Main results: rho, tau per parcel (N=208)
│   ├── mous_band_rho_correlations.csv  # Band-specific rho-DV correlations
│   ├── mous_spectral_confounds.csv     # Spectral confound control results
│   ├── mous_task_stats.csv        # Task replication statistics
│   ├── correlation_stats.csv      # Spatial correlation statistics
│   └── figures/                   # Publication figures
├── analysis/                      # Paper-specific analysis scripts
│   ├── 01_band_specific_analysis.py    # Frequency band decomposition
│   ├── 02_spectral_confounds.py        # Spectral confound control
│   ├── 03_task_replication.py          # Visual/auditory task analysis
│   ├── 04_create_figures.py            # Figure generation
│   └── 05_hcp_yeo17_analysis.py        # HCP dataset replication
├── scripts/                       # MEG processing pipeline
│   ├── 00_extract_tarballs.py     # Data extraction
│   ├── 01_reconall.sh             # FreeSurfer reconstruction
│   ├── 02_make_bem.py             # BEM forward model
│   ├── 03_make_trans.md           # Coregistration instructions
│   ├── 04_extract_parcels_and_metrics.py  # Source reconstruction + metrics
│   ├── 05_group_stats.py          # Group statistics + spin tests
│   └── run_batch.py               # Batch processing
├── meg_axes/                      # Core library
│   ├── config.py                  # Configuration system
│   ├── metrics.py                 # Tau and rho computation
│   ├── preprocessing.py           # Signal preprocessing
│   ├── source.py                  # Source reconstruction
│   └── utils.py                   # Utilities
├── atlas/                         # Parcellation files
│   └── schaefer400_centroids.csv  # Schaefer 400 parcel coordinates
├── config.yaml                    # Main configuration
├── METHODS.md                     # Methods text
└── requirements.txt               # Dependencies
```

## Main Results

### Rho-DV Gradient (Resting State, N=208)

| Band | Correlation (r) | p (spin) |
|------|----------------|----------|
| beta_high | -0.774 | < 0.001 |
| gamma_low | -0.733 | < 0.001 |
| broadband | -0.735 | < 0.001 |
| alpha | -0.649 | < 0.001 |
| delta | +0.550 | < 0.001 |
| theta | +0.466 | < 0.001 |

### Task Replication

| Condition | rho-DV (r) | tau-rho residualized (r) |
|-----------|-----------|-------------------------|
| Rest (N=208) | -0.72 | -0.65 |
| Visual (N=69) | -0.68 | -0.67 |
| Auditory (N=95) | -0.74 | -0.65 |

### Spectral Confound Control

| Model | r | Retention |
|-------|---|-----------|
| Raw rho vs z | -0.72 | 100% |
| rho | gamma_rel | -0.43 | 60% |
| rho | all spectral + RMS | -0.23 | 32% |

## Installation

```bash
# Clone repository
git clone https://github.com/Salardini/Dorso-Ventral-Gradient.git
cd Dorso-Ventral-Gradient

# Create environment
conda create -n meg python=3.10
conda activate meg

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Data

The MOUS MEG dataset is available from the Donders Repository:
- https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236

Processed group-level data for reproducing figures is included in `data/`.

## Metrics

### Rho (Rotational Index)
Computed from delay-embedded VAR(1) dynamics:
1. Embed parcel time series in delay coordinates (d=10, lag=5 samples)
2. Fit VAR(1) model: x(t+1) = A @ x(t)
3. Compute eigenvalues of transition matrix A
4. Rho = mean |Im(lambda)/Re(lambda)| for complex eigenvalue pairs

Higher rho indicates more rotational (oscillatory) dynamics.

### Tau (Intrinsic Timescale)
Autocorrelation integral from 5-300ms lag:
```
tau = integral(ACF(lag), lag_min=5ms, lag_max=300ms)
```

Higher tau indicates longer temporal integration windows.

## Reproducing Analyses

### Band-Specific Analysis
```bash
python analysis/01_band_specific_analysis.py
```
Computes rho in 7 frequency bands (delta through gamma) and tests DV correlations.

### Spectral Confound Control
```bash
python analysis/02_spectral_confounds.py
```
Tests whether rho-DV gradient persists after removing spectral features.

### Task Replication
```bash
python analysis/03_task_replication.py
```
Replicates gradient in visual and auditory task conditions.

### Generate Figures
```bash
python analysis/04_create_figures.py
```

## Citation

```bibtex
@article{salardini2025dorsoventral,
  title={A dorsoventral gradient of rotational dynamics in human cortex},
  author={Salardini, A. and others},
  journal={in preparation},
  year={2025}
}
```

## License

MIT License - see LICENSE file.

## References

- Schaefer et al. (2018) - Local-global parcellation of the human cerebral cortex
- Honey et al. (2012) - Slow cortical dynamics and the accumulation of information
- MNE-Python: https://mne.tools
