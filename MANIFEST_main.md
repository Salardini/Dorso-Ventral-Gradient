# Repository Staging Manifest

Files collected for: "A Dorsoventral Gradient of Rotational Dynamics in Human Cortex"

These are the MAIN MANUSCRIPT files not already in the existing repo (which contains the revision/supplement dual-origin connectivity scripts 01-11).

---

## 1. SCRIPTS

### Core Package: `scripts/main/meg_axes/` — VAR Model & MEG Processing Library

| File | Description |
|------|-------------|
| `__init__.py` | Package init |
| `config.py` | Nested dataclass config system (paths, preprocessing, source, tau/rho params) |
| `metrics.py` | **Core**: Computes tau (ACF integral) and rho (delay-embedded VAR(1) rotational index) per parcel |
| `preprocessing.py` | MEG preprocessing: BIDS loading, notch filter, bandpass, resample, optional ICA |
| `source.py` | Source reconstruction: forward model, dSPM inverse, parcel extraction with PCA-flip |
| `utils.py` | Logging, git versioning, DONE markers, meta.json writing |

### Atlas Package: `scripts/main/atlas/`

| File | Description |
|------|-------------|
| `__init__.py` | Package init |
| `schaefer.py` | Schaefer-400 parcellation loading, fsaverage centroid computation (x=ML, y=AP, z=DV) |

### Pipeline Steps: `scripts/main/pipeline_steps/` — MEG Processing Pipeline

| File | Description |
|------|-------------|
| `00_extract_tarballs.py` | Extract subject tar.gz archives to BIDS structure |
| `01_reconall.sh` | FreeSurfer recon-all for individual subjects |
| `02_make_bem.py` | Create single-shell BEM model from FreeSurfer output |
| `04_extract_parcels_and_metrics.py` | **Core pipeline**: Source reconstruction + Schaefer-400 parcellation + tau/rho computation per subject |
| `05_group_stats.py` | Group-level: mean/median maps, spatial correlations with spin tests, tau-rho correlation |
| `run_batch.py` | Batch runner with dependency management and parallel processing |
| `coreg_check.py` | Verify MEG-MRI coregistration quality |
| `batch_process.sh` / `batch_process_v2.sh` | Shell batch processing scripts |
| `smoke_test_one_subject.sh` | Quick single-subject pipeline test |

### Analysis Scripts: `scripts/main/analysis/` — DVG Post-Hoc Analyses

| File | Description |
|------|-------------|
| `01_compute_spatial_correlations.py` | Rho/tau vs spatial coordinates (DV, AP, ML) with spin permutation tests |
| `02_spectral_confounds.py` | Control for spectral exponent, band power confounds on rho-DV gradient |
| `03_task_replication.py` | Replicate rho-DV gradient in auditory/visual task conditions |
| `04_principal_gradient_comparison.py` | Compare rho gradient with Margulies principal gradient |
| `05_generate_figures.py` | Generate main manuscript figures |
| `06_nonlinear_validation.py` | Test for nonlinear rho-DV relationships |
| `07_gradient_axis_angles.py` | Compute angle between rho gradient and canonical brain axes |
| `08_tau_rho_regional.py` | Regional tau-rho relationship analysis |
| `09_critical_spectral_test.py` | Critical test: does spectral exponent explain away rho-DV? |
| `10_split_half_reliability.py` | Split-half reliability of rho/tau maps |
| `11_internal_consistency.py` | Internal consistency of parcel metrics |
| `12_depth_bias_control.py` | Control for MEG depth bias on cortical surface |
| `13_axis_angle_validation.py` | Validate that DV gradient is not an artifact of axis orientation |
| `14_spectral_features.py` | Compute spectral features per parcel |
| `supplementary/01_band_specific_analysis.py` | Band-specific (delta through gamma) rho analysis |
| `supplementary/05_hcp_yeo17_analysis.py` | HCP MEG Yeo17 parcellation analysis |
| `utils/__init__.py` | Analysis utils package |
| `utils/plotting.py` | Shared plotting functions |
| `utils/spin_test.py` | Spin permutation test implementation |

### Standalone Analysis Scripts: `scripts/main/`

| File | Description |
|------|-------------|
| `adaptive_delay_analysis.py` | **Adaptive delay**: Tests if band-specific rho gradients are artifacts of fixed embedding delay |
| `atlas_sensitivity_analysis.py` | **Atlas sensitivity**: Tests rho-DV gradient at Schaefer 100/200/400 resolutions |
| `compute_spectral_features_v2.py` | Spectral exponent (1/f slope), band powers, peak frequency per parcel |
| `fmri_replication_pipeline_v2.py` | **MOUS fMRI replication**: Computes rho/tau/spectral from fMRI timeseries |
| `hcp_rho_replication.py` | **HCP fMRI replication**: ~1000 subjects, Schaefer-400, resting-state |
| `hcp_replication_figure.py` | Generate HCP replication figure |
| `hcp_rho_task.py` | HCP task-fMRI rho analysis |
| `06_hcp_tau_rho.py` | HCP MEG tau/rho from CIFTI ptseries (Yeo parcellation) |
| `compute_yeo17_centroids.py` | Compute Yeo17 split-label centroids for HCP MEG data |
| `computational_model_tau_rho.py` | **E-I network model**: Demonstrates tau-rho tradeoff from varying inhibitory gain |
| `ahba_gene_expression_analysis.py` | AHBA gene expression: PV/SST interneuron markers vs rho gradient |
| `tau_rho_model.py` | Simplified tau-rho tradeoff model (parameter sweep) |
| `generate_all_figures.py` | Master figure generator (all main + supplementary figures) |
| `generate_fig1b_scatter.py` | Generate Figure 1b scatter plot |
| `paper1_final_analyses.py` | Final statistics: rho vs principal gradient, R2 distributions, summary table |
| `split_half_stability.py` | Split-half stability of subject-level rho-DV correlations |
| `split_half_from_correlations.py` | Split-half analysis from pre-computed correlations |

---

## 2. DATA FILES

### `data/main/adaptive_delay_analysis/`
- `adaptive_delay_correlations.csv` — Band-specific rho-DV correlations with adaptive vs fixed delay
- `adaptive_delay_parcel_results.csv` — Per-parcel adaptive delay rho values

### `data/main/atlas_sensitivity/`
- `atlas_sensitivity_correlations.csv` — Rho-DV correlations at 100/200/400 resolutions
- `atlas_sensitivity_parcel_results.csv` — Per-parcel results across resolutions
- `parcel_means_schaefer100.csv` / `200` / `400` — Parcel means at each resolution

### `data/main/fmri_replication/`
- `fmri_replication_results.csv` / `_v2.csv` — MOUS fMRI rho-DV correlation results
- `fmri_parcel_measures.csv` / `_v2.csv` — Per-parcel fMRI-derived rho/tau/spectral measures

### `data/main/fMRI2_results/`
- `hcp_parcel_rho.csv` — HCP fMRI rho per Schaefer-400 parcel
- `hcp_replication_summary.csv` — Summary of HCP rho-DV replication
- `hcp_task_parcel_rho.csv` — HCP task-fMRI rho per parcel
- `hcp_task_summary.csv` — Summary of HCP task rho-DV results

### `data/main/group/`
- `parcel_group_maps.csv` — Group-average rho/tau maps (400 parcels)
- `correlation_stats.csv` — All spatial correlations with p-values
- `qc_summary.csv` — QC metrics across subjects
- `mous_parcel_with_spectral.csv` — Parcel maps merged with spectral features
- `mous_spectral_confounds.csv` — Spectral confound control results
- `mous_auditory_task_group.csv` / `mous_visual_task_group.csv` — Task condition group maps
- `mous_task_stats.csv` — Task replication statistics

### `data/main/HCP/`
- `hcp_group_stats.csv` — HCP MEG group-level statistics
- `hcp_correlation_stats.csv` — HCP rho-DV correlations with spin tests
- `hcp_subject_metrics.csv` — Per-subject HCP MEG metrics
- `yeo17_split_centroids.csv` — Yeo17 split-label centroids
- `correlation_stats.csv` — DVG-computed HCP correlations
- `spin_test_gammalow.csv` — Gamma-low band spin test results
- `hcp_meta.json` — HCP analysis metadata

### `data/main/mous/`
- `parcel_group_maps.csv` — MOUS group-average parcel metrics
- `correlation_stats.csv` — MOUS spatial correlations
- `spectral_confounds.csv` — Spectral confound analysis
- `gradient_axis_angles.csv` — Gradient vs axis angles
- `band_rho_correlations.csv` — Band-specific rho-DV correlations
- `qc_summary.csv` — MOUS QC
- `auditory_task_group.csv` / `visual_task_group.csv` — Task condition data
- `task_stats.csv` — Task replication statistics

### `data/main/validation/`
- `nonlinear_validation.csv` / `_correlations.csv` / `_summary.csv` — Nonlinear rho-DV tests

### `data/main/` (root)
- `ahba_rho_correlations.csv` — AHBA gene expression vs rho correlations
- `ahba_rho_merged.csv` — Merged AHBA + rho dataset
- `schaefer400_all_subjects.csv` — All subjects' parcel-level data (subject-level analysis)
- `schaefer400_parcel_means.csv` — Group parcel means
- `regional_summary.csv` / `regional_timescales.csv` — Regional summary statistics

---

## 3. FIGURES

### `figures/main/` — Main Manuscript Figures

**Named figure series (from DVG analysis):**
- `fig1b_rho_vs_dv_scatter.{png,pdf}` — Rho vs DV coordinate scatter
- `fig2_rho_dv_gradient.png` — Rho dorsoventral gradient brain map
- `fig2_spectral_tradeoff.png` — Spectral frequency tradeoff
- `fig3_embedding_delay_robustness.png` — Embedding delay robustness
- `fig3_task_replication.png` — Task replication
- `fig4_orthogonality.png` — Rho-tau orthogonality
- `fig4_tau_rho_relationship.png` — Tau-rho relationship
- `fig5_fmri_replication.png` — fMRI replication
- `fig_ahba_pv_sst.{png,pdf}` — AHBA PV/SST gene expression
- `fig_ahba_results.{png,pdf}` — AHBA full results
- `fig_computational_model.{png,pdf}` — E-I computational model
- `fig_model_ahba_ABC.{png,pdf}` — Combined model + AHBA panel figure

**Publication-numbered figures:**
- `Figure1_gradient.{png,pdf}` — Main Figure 1: Gradient overview
- `Figure2_frequency.{png,pdf}` — Main Figure 2: Frequency analysis
- `FigureS1_parcellation.{png,pdf}` through `FigureS6_networks.{png,pdf}` — Supplementary figures

**Cross-validation figures:**
- `fig1_network_analysis.{png,pdf}` — Network analysis
- `fig2_spatial_gradients.{png,pdf}` — Spatial gradients
- `fig3_cross_species.{png,pdf}` — Cross-species comparison
- `hcp_replication_figure.{png,pdf}` — HCP replication

### `figures/main/extended/`
- Extended/supplementary QC figures (spin test, model fit, reliability)

---

## 4. CONFIG

- `config.yaml` — Main pipeline configuration (paths, preprocessing, source, tau/rho params)
- `config_ds004998.yaml` — Config for ds004998 dataset
- `config_visual.yaml` — Config for visual task analysis
- `requirements.txt` — Python dependencies
- `setup.py` — Package installation

---

## Coverage by Requested Analysis

| Analysis | Scripts Found | Data Found |
|----------|--------------|------------|
| 1. VAR model fitting | meg_axes/metrics.py, pipeline_steps/04_extract_parcels_and_metrics.py | group/parcel_group_maps.csv, mous/* |
| 2. MEG preprocessing | meg_axes/preprocessing.py, meg_axes/source.py, pipeline_steps/00-05 | (raw data not staged — too large) |
| 3. Spectral analysis | compute_spectral_features_v2.py, analysis/14_spectral_features.py | group/mous_parcel_with_spectral.csv, mous/spectral_confounds.csv |
| 4. Atlas sensitivity | atlas_sensitivity_analysis.py | atlas_sensitivity/*.csv |
| 5. fMRI replication | fmri_replication_pipeline_v2.py, hcp_rho_replication.py | fmri_replication/*.csv, fMRI2_results/*.csv |
| 6. Adaptive delay | adaptive_delay_analysis.py | adaptive_delay_analysis/*.csv |
| 7. Subject-level | pipeline_steps/05_group_stats.py, paper1_final_analyses.py | schaefer400_all_subjects.csv, HCP/hcp_group_stats.csv, HCP/hcp_correlation_stats.csv |
| 8. Main figures | generate_all_figures.py, analysis/05_generate_figures.py | figures/main/*.{png,pdf} |
| 9. Yeo17 split | compute_yeo17_centroids.py, supplementary/05_hcp_yeo17_analysis.py | HCP/yeo17_split_centroids.csv |
