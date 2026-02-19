# Architectural Predictors of the Cortical Rotational Dynamics Gradient

Code and data for the dual-origin connectivity analysis in *"A Dorsoventral Gradient of Rotational Dynamics in Human Cortex"*.

## Summary

The cortical ρ metric (rotational dynamics derived from VAR models of MEG resting-state data) exhibits a dorsoventral gradient (ρₛ = −0.73, p < 10⁻⁶⁰) across Schaefer 400 parcels. This repository contains all code and supporting data for identifying the architectural predictors of this gradient, centered on the finding that **the dual-origin structural covariance gradient (G2; Valk et al., 2020)** is the only predictor that survives comprehensive controls for spatial position, spectral properties, and intrinsic timescales (r|z+SE+τ = −0.249, p_spin = 0.018).

## Key Results

| Predictor | ρₛ | r\|z | r\|z+SE+τ | p_spin |
|---|---|---|---|---|
| z-coordinate | −0.729 | — | — | — |
| **SCov G2 (dual origin)** | **−0.155** | **−0.213** | **−0.249** | **0.018** |
| **Genetic G2** | **−0.153** | **−0.198** | **−0.231** | **0.020** |
| SC short/long ratio | −0.273 | −0.158 | −0.115 | — |
| BigBrain MPC G1 | −0.034 | +0.025 | +0.003 | — |
| Mesulam type | −0.027 | −0.038 | — | — |
| PVALB expression | −0.073 | −0.012 | — | — |
| SST expression | +0.124 | +0.013 | — | — |

**Variance explained:** z alone R² = 0.523; z + SE: R² = 0.868; z + SE + τ: R² = 0.870; z + SE + τ + G2: R² = 0.878.

## Repository Structure

```
├── data/
│   ├── rho_master.csv                          # Master dataset (400 parcels × 97 columns)
│   ├── glasser_360_conte69.csv                 # Glasser vertex labels on conte69
│   ├── schaefer_400_conte69.csv                # Schaefer vertex labels on conte69
│   ├── mesulam_conte69.csv                     # Mesulam vertex labels on conte69
│   ├── strcov.csv, strcov_gradient.csv         # Valk structural covariance data
│   ├── coher.csv, coher_gradient.csv           # Valk genetic correlation data
│   ├── BigBrain_intensity_profiles_glasser.txt # BigBrain laminar profiles (15×361)
│   ├── BigBrain_MPC_glasser.txt                # BigBrain MPC matrix (361×361)
│   ├── HCP_intensity_profiles_glasser.txt      # HCP MPC profiles
│   ├── HCP_MPC_glasser.txt                     # HCP MPC matrix
│   ├── parcel_spectral_features.csv            # Spectral exponent, peak freq, etc.
│   └── parcel_group_maps.csv                   # Group-level ρ, τ, coordinates
├── scripts/
│   ├── 01_gene_expression.py                   # AHBA PVALB/SST analysis
│   ├── 02_fc_metrics.py                        # FC gradients, graph metrics, Mesulam
│   ├── 03_valk_gradients.py                    # Primary G2 analysis
│   ├── 04_bigbrain_mpc.py                      # BigBrain MPC (original KMeans mapping)
│   ├── 05_comprehensive_figure.py              # Exploratory multi-panel figure
│   ├── 06_spectral_tau_controls.py             # Progressive partial correlations
│   ├── 07_spin_tests.py                        # 5,000 spin permutation tests
│   ├── 08_structural_connectivity.py           # HCP SC from ENIGMA Toolbox
│   ├── 09_bigbrain_proper.py                   # BigBrain with proper Glasser mapping
│   ├── 10_publication_figure.py                # Main publication figure
│   └── 11_brain_surface_plots.py               # Surface renderings
├── figures/
│   ├── fig_pub_v5.pdf                          # Main 4-panel figure
│   ├── fig_pub_sc.pdf                          # SC analysis 6-panel figure
│   └── fig_spin_tests.pdf                      # Spin test null distributions
├── writing/
│   ├── methods_final.md                        # Complete methods section
│   ├── discussion_final.md                     # Complete discussion section
│   ├── bigbrain_methods_results.md             # BigBrain analysis write-up
│   ├── sc_methods_results.md                   # SC analysis write-up
│   ├── spectral_exponent_summary.md            # SE control summary
│   └── tau_analysis_summary.md                 # τ analysis summary
└── README.md
```

## Requirements

```
numpy scipy pandas scikit-learn matplotlib
brainspace          # Cortical gradients, parcellations, surface data
abagen              # Allen Human Brain Atlas gene expression
enigmatoolbox       # HCP structural connectivity (pip install git+https://github.com/MICA-MNI/ENIGMA.git)
```

## Data Sources

| Resource | Source | Used in |
|---|---|---|
| Valk et al. (2020) structural covariance gradients | [GitHub](https://github.com/sofievalk/projects) | Script 03 |
| HCP group-average FC (Schaefer 400) | BrainSpace `load_group_fc()` | Script 02 |
| HCP structural connectivity (Schaefer 400) | ENIGMA Toolbox `load_sc()` | Script 08 |
| Allen Human Brain Atlas | abagen toolbox | Script 01 |
| Mesulam classification | BrainSpace `mesulam_conte69.csv` | Script 02 |
| BigBrain intensity profiles and MPC (Glasser 360) | [MICA-MNI/micaopen](https://github.com/MICA-MNI/micaopen) | Scripts 04, 09 |
| Glasser 360 conte69 labels | ENIGMA Toolbox | Script 09 |
| Schaefer 400 atlas | [CBIG](https://github.com/ThomasYeoLab/CBIG) | All |

## References

- Valk, S. L., et al. (2020). Shaping brain structure: Genetic and phylogenetic axes of macroscale organization of cortical thickness. *Science Advances,* 6, eabb3417.
- Paquola, C., et al. (2019). Microstructural and functional gradients are increasingly dissociated in transmodal cortices. *PLOS Biology,* 17, e3000284.
- Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage,* 178, 540–551.
- Larivière, S., et al. (2021). The ENIGMA Toolbox. *Nature Methods,* 18, 698–700.
- Pandya, D. N., et al. (2015). *Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept.* Oxford University Press.
