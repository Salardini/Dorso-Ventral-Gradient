# A Dorsoventral Gradient of Rotational Dynamics in Human Cortex

Code and data for reproducing analyses from:

**Salardini A., et al. "A dorsoventral gradient of rotational dynamics in human cortex."** *(in preparation)*

## Abstract

Neural dynamics vary systematically across the cortical sheet. We demonstrate that
rotational dynamics (rho) exhibit a robust gradient along the dorsoventral axis
(r = -0.72, p_spin = 0.002), forming a near-vertical axis (18 deg from DV) that is
orthogonal to the principal gradient of functional connectivity (r = -0.04).
This gradient replicates across cognitive states and is independently validated
by permutation entropy (r = -0.57).

## Key Findings

| Finding | Statistic |
|---------|-----------|
| rho-DV (MOUS rest) | r = -0.72, p_spin = 0.002 |
| rho-DV (Visual task) | r = -0.68, p_spin < 0.001 |
| rho-DV (Auditory task) | r = -0.74, p_spin < 0.001 |
| rho vs Principal Gradient | r = -0.04 (orthogonal) |
| After spectral control | r = -0.23, p_spin < 0.001 |
| tau-rho residualized | r = -0.67 |
| Gradient axis angle | 18 deg from dorsoventral |
| Permutation entropy validation | r = -0.57 (same DV gradient) |

## Repository Structure

```
Dorso-Ventral-Gradient/
├── README.md                    # This file
├── METHODS.md                   # Detailed methods text
├── requirements.txt             # Python dependencies
├── config.yaml                  # Pipeline configuration
│
├── atlas/                       # Parcellation files
│   └── schaefer400_centroids.csv
│
├── scripts/                     # MEG processing pipeline
│   ├── 00_extract_tarballs.py
│   ├── 01_reconall.sh
│   ├── 02_make_bem.py
│   ├── 03_make_trans.md
│   ├── 04_extract_parcels_and_metrics.py
│   └── 05_group_stats.py
│
├── analysis/                    # Paper 1 analysis scripts
│   ├── 01_compute_spatial_correlations.py
│   ├── 02_spectral_confounds.py
│   ├── 03_task_replication.py
│   ├── 04_principal_gradient_comparison.py
│   ├── 05_generate_figures.py
│   ├── 06_nonlinear_validation.py
│   ├── 07_gradient_axis_angles.py
│   ├── 08_tau_rho_regional.py
│   ├── utils/
│   │   ├── spin_test.py
│   │   └── plotting.py
│   └── supplementary/
│       ├── 01_band_specific_analysis.py
│       └── 05_hcp_yeo17_analysis.py
│
├── data/                        # Processed data
│   ├── mous/
│   │   ├── parcel_group_maps.csv       # Main results (N=208)
│   │   ├── spectral_confounds.csv      # Confound control
│   │   ├── task_stats.csv              # Task replication stats
│   │   ├── visual_task_group.csv       # Visual task data (N=69)
│   │   ├── auditory_task_group.csv     # Auditory task data (N=95)
│   │   ├── band_rho_correlations.csv   # Frequency band analysis
│   │   ├── gradient_axis_angles.csv    # 3D gradient directions
│   │   └── qc_summary.csv              # Quality control
│   ├── hcp/
│   │   ├── correlation_stats.csv       # HCP correlations
│   │   └── spin_test_gammalow.csv      # HCP spin test
│   ├── validation/
│   │   ├── nonlinear_validation.csv
│   │   ├── nonlinear_validation_summary.csv
│   │   └── nonlinear_validation_correlations.csv
│   └── reference/
│       └── margulies_pc1_schaefer400.csv  # Principal gradient
│
├── figures/                     # Publication figures
│   ├── fig2_rho_dv_gradient.png
│   ├── fig3_task_replication.png
│   ├── fig4_orthogonality.png
│   └── extended/
│       ├── qc_model_fit.png
│       ├── hcp_spin_test.png
│       └── tau_vs_rho_residualized.png
│
├── results/                     # Analysis outputs
│   ├── spatial_correlations.csv
│   └── pc1_comparison.csv
│
└── meg_axes/                    # Core library
    ├── config.py
    ├── metrics.py
    ├── preprocessing.py
    ├── source.py
    └── utils.py
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
| Rest (N=208) | -0.72 | -0.67 |
| Visual (N=69) | -0.68 | -0.67 |
| Auditory (N=95) | -0.74 | -0.65 |

### Spectral Confound Control

| Model | r | Retention |
|-------|---|-----------|
| Raw rho vs z | -0.72 | 100% |
| rho | gamma_rel | -0.43 | 60% |
| rho | all spectral + RMS | -0.23 | 32% |

### Gradient Axis Angles

| Metric | Region | Angle from DV | Dominant Axis |
|--------|--------|---------------|---------------|
| rho | Full cortex | 18 deg | DV |
| rho | Frontal | 17 deg | DV |
| rho | Posterior | 13 deg | DV |
| tau | Frontal | 6 deg | DV |
| tau | Posterior | 75 deg | AP |

### Nonlinear Validation

| Measure | r with rho | r with z (DV) | Validates rho? |
|---------|----------|---------------|--------------|
| Spectral Entropy | +0.28 | +0.08 | No |
| Sample Entropy | +0.16 | +0.18 | No |
| **Permutation Entropy** | **+0.49** | **-0.57** | **Yes** |

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

## Reproduce Analyses

```bash
cd analysis

# 1. Spatial correlations (rho-DV, tau-AP)
python 01_compute_spatial_correlations.py

# 2. Spectral confound control
python 02_spectral_confounds.py

# 3. Task replication (visual/auditory)
python 03_task_replication.py

# 4. Principal gradient comparison
python 04_principal_gradient_comparison.py

# 5. Generate figures
python 05_generate_figures.py

# 6. Nonlinear validation (entropy measures)
python 06_nonlinear_validation.py

# 7. Gradient axis angles
python 07_gradient_axis_angles.py

# 8. Regional tau-rho analysis
python 08_tau_rho_regional.py
```

## Key Statistics

| Analysis | Value | Source File |
|----------|-------|-------------|
| rho-DV rest | r = -0.72, p_spin = 0.002 | parcel_group_maps.csv |
| rho-DV visual | r = -0.68, p_spin < 0.001 | visual_task_group.csv |
| rho-DV auditory | r = -0.74, p_spin < 0.001 | auditory_task_group.csv |
| rho vs PC1 | r = -0.04, p = 0.45 | + margulies_pc1.csv |
| tau-rho raw | r ~ 0 | parcel_group_maps.csv |
| tau-rho residualized | r = -0.67 | parcel_group_maps.csv |
| rho-DV after gamma | r = -0.43 | spectral_confounds.csv |
| rho-DV after all | r = -0.23 | spectral_confounds.csv |
| HCP gamma-low | r = -0.40, p_spin = 0.12 | hcp/correlation_stats.csv |
| rho axis angle | 18 deg from DV | gradient_axis_angles.csv |
| rho vs perm entropy | r = +0.49 | nonlinear_validation_summary.csv |
| Perm entropy DV | r = -0.57 | nonlinear_validation_summary.csv |
| Frontal tau-rho raw | r = +0.26 | regional analysis |
| Posterior tau-rho raw | r = -0.45 | regional analysis |

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
- Margulies et al. (2016) - Situating the default-mode network
- MNE-Python: https://mne.tools
