# Methods

## MEG Data Acquisition and Preprocessing

### Dataset
Magnetoencephalography (MEG) data were acquired from the Mother of Unification Studies (MOUS) dataset, comprising resting-state recordings from N subjects. Data were collected using a 275-channel CTF MEG system at a sampling rate of 1200 Hz.

### Preprocessing Pipeline
MEG data were preprocessed using MNE-Python (version 1.x). The preprocessing pipeline consisted of:

1. **Line noise removal**: Notch filtering at 60 Hz and harmonics (120, 180 Hz) to remove power line interference.

2. **Bandpass filtering**: Data were bandpass filtered between 1-40 Hz using a zero-phase FIR filter to isolate neural oscillations while removing slow drifts and high-frequency noise.

3. **Resampling**: Data were downsampled to 200 Hz to reduce computational burden while preserving relevant frequency content up to 40 Hz (Nyquist = 100 Hz).

4. **Channel selection**: Only MEG gradiometer and magnetometer channels were retained; reference channels, EEG, EOG, and ECG channels were excluded.

5. **Duration matching**: For task recordings, data were truncated to 300 seconds to match resting-state recording duration.

## Source Reconstruction

### Forward Modeling
Source reconstruction employed a template-based approach using the fsaverage brain from FreeSurfer. This approach enables processing without subject-specific MRI data while providing anatomically-informed source localization.

A boundary element model (BEM) was constructed using a single-shell conductor model (conductivity = 0.3 S/m) appropriate for MEG. The source space was defined on the cortical surface using an octahedral subdivision (oct6 spacing, ~4 mm resolution), yielding approximately 20,484 source locations distributed across both hemispheres.

### Inverse Solution
Source activity was estimated using dynamic statistical parametric mapping (dSPM; Dale et al., 2000). The noise covariance matrix was computed empirically from the preprocessed data. The inverse operator was constructed with the following parameters:
- **SNR assumption**: 3.0 (λ² = 1/9)
- **Loose orientation constraint**: 0.2 (allowing moderate deviation from cortical normal)
- **Depth weighting**: 0.8 (compensating for depth bias in MEG sensitivity)

Source orientations were constrained to be predominantly normal to the cortical surface (pick_ori='normal'), yielding a scalar time series per source location.

## Cortical Parcellation

### Schaefer Atlas
Source-level activity was parcellated using the Schaefer 400-parcel atlas (Schaefer et al., 2018), which provides a functionally-derived parcellation organized into 7 canonical resting-state networks (Yeo et al., 2011):
- Visual
- Somatomotor
- Dorsal Attention
- Ventral Attention
- Limbic
- Frontoparietal Control
- Default Mode

### Time Series Extraction
Parcel-level time series were extracted using PCA-based sign-flipping (mode='pca_flip'), which:
1. Extracts the first principal component across all vertices within each parcel
2. Applies sign correction to ensure consistent orientation across subjects
3. Yields a single representative time series per parcel

This approach maximizes explained variance while maintaining interpretability and cross-subject comparability.

## Temporal Dynamics Metrics

### Intrinsic Neural Timescale (τ)

The intrinsic neural timescale quantifies the temporal integration window of neural activity, reflecting how slowly local activity decorrelates over time (Murray et al., 2014; Honey et al., 2012).

**Computation method (ACF integral)**:
1. Compute the autocorrelation function (ACF) using FFT-based convolution for computational efficiency
2. Integrate the positive portion of the ACF from lag_min (5 ms) to lag_max (300 ms):

$$\tau = \int_{t_{min}}^{t_{max}} \max(ACF(t), 0) \, dt$$

The integration bounds were chosen to capture timescales relevant to neural computation while excluding very short lags dominated by measurement noise and very long lags where the ACF approaches zero.

**Secondary estimator (exponential fit)**:
As a robustness check, τ was also estimated by fitting an exponential decay model to the ACF:

$$ACF(t) = e^{-t/\tau}$$

The fit quality was assessed using R² and RMSE metrics.

### Rotational Dynamics Index (ρ)

The rotational dynamics index characterizes the oscillatory versus decaying nature of neural dynamics using delay embedding and linear dynamical systems analysis.

**Computation method**:
1. **Delay embedding**: Construct a delay-embedded state space representation:
   - Embedding dimension: m = 10
   - Delay: d = 1 sample (5 ms at 200 Hz)
   - This creates an m-dimensional trajectory from the scalar time series

2. **VAR(1) model fitting**: Fit a first-order vector autoregressive model with ridge regularization (α = 0.001):
$$\mathbf{x}(t+1) = \mathbf{A} \cdot \mathbf{x}(t) + \boldsymbol{\epsilon}(t)$$

3. **Eigenvalue analysis**: Compute eigenvalues λ of the transition matrix A

4. **Rotational index**: Calculate the mean ratio of imaginary to total magnitude for eigenvalues with |λ| > 0.01:
$$\rho = \frac{1}{n} \sum_{i} \frac{|\text{Im}(\lambda_i)|}{|\lambda_i|}$$

Higher ρ values indicate more rotational/oscillatory dynamics (complex eigenvalues), while lower values indicate predominantly decaying dynamics (real eigenvalues).

**Quality control**: The one-step prediction R² of the VAR(1) model was computed to assess fit quality.

## Group-Level Statistical Analysis

### Spatial Coordinate System
Parcel coordinates were defined in fsaverage surface space using RAS (Right-Anterior-Superior) orientation:
- **x**: Medial-Lateral (ML) axis (negative = left, positive = right)
- **y**: Anterior-Posterior (AP) axis (negative = posterior, positive = anterior)
- **z**: Dorsal-Ventral (DV) axis (negative = inferior, positive = superior)

Parcel centroids were computed as the mean coordinate of all vertices within each parcel.

### Correlation Analysis
Spatial relationships between temporal metrics (τ, ρ) and anatomical axes were assessed using Spearman rank correlation, which is robust to outliers and does not assume linearity.

### Spin Permutation Testing
Statistical significance was assessed using spin permutation tests (Alexander-Bloch et al., 2018), which preserve the spatial autocorrelation structure of cortical maps. This approach:
1. Rotates one cortical map on the spherical surface using random rotations
2. Recomputes the correlation for each rotation
3. Derives a null distribution that accounts for spatial smoothness

The two-tailed p-value was computed as:
$$p = \frac{\sum_{i=1}^{n_{perm}} \mathbb{1}(|r_{null,i}| \geq |r_{obs}|) + 1}{n_{perm} + 1}$$

We used 1,000 permutations for all analyses. When spin tests failed (e.g., due to coordinate issues), we fell back to standard (non-spatial) permutation testing.

### Hierarchical Robustness Analysis
To assess whether the τ-ρ relationship is independent of spatial location and signal quality, we performed hierarchical regression analyses:
1. **Raw correlation**: τ vs ρ without covariates
2. **Residualized on coordinates**: τ and ρ residualized on (x, y, z) coordinates
3. **Residualized on coordinates + variance**: Additionally controlling for time series variance
4. **Residualized on coordinates + variance + RMS**: Full model controlling for all potential confounds

### Quality Control Metrics
For each parcel, we computed:
- **ts_var**: Time series variance (signal strength proxy)
- **ts_rms**: Root mean square amplitude (SNR proxy)
- **tau_exp_r2**: R² of exponential ACF fit (τ reliability)
- **rho_r2**: R² of VAR(1) one-step prediction (ρ reliability)

## Software and Reproducibility

All analyses were performed using:
- **Python** 3.10+
- **MNE-Python** 1.x (MEG processing and source reconstruction)
- **NumPy** 1.x (numerical computing)
- **SciPy** 1.x (signal processing, optimization)
- **Pandas** 2.x (data management)
- **scikit-learn** 1.x (regression)
- **BrainSpace** 0.x (spin permutation tests)
- **Matplotlib** 3.x (visualization)

Complete processing metadata, including software versions, parameters, and git commit hashes, were saved in JSON format for each subject and group analysis to ensure full reproducibility.

## References

Alexander-Bloch, A. F., Shou, H., Liu, S., Satterthwaite, T. D., Glahn, D. C., Shinohara, R. T., ... & Raznahan, A. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage*, 178, 540-551.

Dale, A. M., Liu, A. K., Fischl, B. R., Buckner, R. L., Belliveau, J. W., Lewine, J. D., & Halgren, E. (2000). Dynamic statistical parametric mapping: combining fMRI and MEG for high-resolution imaging of cortical activity. *Neuron*, 26(1), 55-67.

Honey, C. J., Thesen, T., Donner, T. H., Silbert, L. J., Carlson, C. E., Devinsky, O., ... & Hasson, U. (2012). Slow cortical dynamics and the accumulation of information over long timescales. *Neuron*, 76(2), 423-434.

Murray, J. D., Bernacchia, A., Freedman, D. J., Romo, R., Wallis, J. D., Cai, X., ... & Wang, X. J. (2014). A hierarchy of intrinsic timescales across primate cortex. *Nature Neuroscience*, 17(12), 1661-1663.

Schaefer, A., Kong, R., Gordon, E. M., Laumann, T. O., Zuo, X. N., Holmes, A. J., ... & Yeo, B. T. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex*, 28(9), 3095-3114.

Yeo, B. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D., Hollinshead, M., ... & Buckner, R. L. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology*, 106(3), 1125-1165.
