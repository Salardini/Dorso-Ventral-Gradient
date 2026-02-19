# Methods: Architectural Predictors of the Rotational Dynamics Gradient

## Overview

To identify the structural and functional correlates of the cortical ρ gradient, we performed a systematic evaluation of candidate predictors spanning developmental structural gradients, functional connectivity topology, cytoarchitectural classification, interneuron gene expression, spectral properties, and intrinsic timescales. All analyses used the Schaefer 400 × 7 networks parcellation (Schaefer et al., 2018), matching the parcellation used for ρ estimation. Analyses were performed in Python 3.12 using NumPy, SciPy, pandas, scikit-learn, and BrainSpace (Vos de Wael et al., 2020).

## Parcel-level measures

MNI centroid coordinates (x, y, z) and Yeo 7-network assignments were obtained for each of the 400 parcels. The z-coordinate served as the dorsoventral spatial axis. The MEG-derived spectral exponent (1/f slope) and intrinsic timescale (τ, the autocorrelation decay constant from the VAR model) were computed per parcel as described in the main text.

## Structural covariance gradients

Cortical thickness structural covariance and genetic correlation gradient values were obtained from Valk et al. (2020; https://github.com/sofievalk/projects). Structural covariance gradients were computed via diffusion map embedding of the group-average cortical thickness covariance matrix from HCP participants, parcellated to Schaefer 400. The second gradient (G2) captures an organizational axis that aligns with the dual origin model of cortical development, correlating with geodesic distance from paleocortex (r = 0.67, p_spin < 0.001; Valk et al., 2020). Genetic correlation gradients, derived from twin-based heritability analyses of cortical thickness in HCP, were used to confirm the heritability of the structural gradient (genetic correlation between phenotypic and genetic G2: r = 0.96).

## Functional connectivity metrics

Group-average resting-state functional connectivity was obtained from HCP S1200 data, parcellated to Schaefer 400 via BrainSpace. FC gradients were computed using diffusion map embedding (cosine kernel, α = 0.5, 10 components) applied to the thresholded FC matrix (top 10% of connections retained). FC Gradient 1 captures the principal sensorimotor-to-default mode hierarchy (Margulies et al., 2016). The FC matrix was binarized at the 90th percentile to compute graph-theoretic metrics: clustering coefficient (local triangle density) and participation coefficient (between-network connectivity diversity, computed with respect to Yeo 7-network assignments).

## Cytoarchitectural classification

Mesulam cytoarchitectural type (1 = idiotypic/primary, 2 = unimodal, 3 = heteromodal, 4 = paralimbic) was assigned to each Schaefer 400 parcel using vertex-level Mesulam labels on the conte69 32k surface (BrainSpace). Each parcel was assigned the modal Mesulam label across its constituent vertices.

## Interneuron gene expression

Parcel-level expression of PVALB (parvalbumin, marking fast-spiking PV+ interneurons) and SST (somatostatin, marking SST+ interneurons) was obtained from the Allen Human Brain Atlas (AHBA; Hawrylycz et al., 2012) via the abagen toolbox (Markello et al., 2021). Expression values were parcellated to Schaefer 400 using default abagen parameters (donor normalization via scaled robust sigmoid, inter-areal differential stability filtering). Z-scored expression values were averaged across available donors (n = 6 for left hemisphere, n = 2 for bilateral coverage). PVALB was available for 345/400 parcels; SST for 347/400 parcels.

## BigBrain cytoarchitectural measures

Histological laminar features were obtained from the BigBrain atlas (Amunts et al., 2013) via the MICA laboratory's open data (Paquola et al., 2019). Staining intensity profiles (15 equivolumetric intracortical surfaces) and a microstructure profile covariance (MPC) matrix were available for 360 Glasser atlas parcels (Glasser et al., 2016). From the profiles we computed per-parcel SD (laminar differentiation), CV, skewness, kurtosis, and mean absolute gradient. MPC Gradient 1 was derived via diffusion map embedding (normalized angle kernel; BrainSpace) and captures the principal sensory-to-limbic cytoarchitectural axis. MPC node strength (mean covariance per parcel) was also computed.

To map BigBrain features from Glasser 360 to Schaefer 400, we obtained vertex-level Glasser parcellation labels on the conte69 32k surface (ENIGMA Toolbox; Larivière et al., 2021). For each Schaefer parcel, the modal Glasser label across its constituent vertices was assigned (mean modal overlap fraction = 0.59; 400/400 parcels mapped). BigBrain feature values for each Schaefer parcel were taken from its assigned Glasser parcel.

## Structural connectivity

Group-average structural connectivity was obtained from HCP diffusion-weighted MRI via the ENIGMA Toolbox (Larivière et al., 2021), parcellated to Schaefer 400. The 400 × 400 SC matrix represents log-transformed streamline counts from deterministic tractography, averaged across HCP participants. Parcel labels were reindexed to match the analysis order using exact string matching. From the SC matrix we computed per-parcel node strength, degree, weighted mean connection distance (using MNI centroid coordinates), short-range strength (connections < 50 mm), long-range strength (≥ 50 mm), short/long range ratio, weighted clustering coefficient (Onnela et al., 2005), participation coefficient (with respect to Yeo 7-network assignments), and within-network connectivity fraction.

## Statistical analyses

All bivariate associations report Spearman rank correlations (ρₛ). To assess whether each predictor explains ρ variance beyond the dorsoventral axis and other confounds, we computed partial Spearman correlations using a rank-and-residualize procedure: all variables were rank-transformed, the ranks of ρ and the predictor were separately regressed on the ranks of the control variables (z-coordinate, spectral exponent, and/or τ) via ordinary least squares, and Pearson r was computed between the two sets of residuals. We report three levels of control: (i) controlling for z alone, to remove the dominant spatial gradient; (ii) controlling for z and spectral exponent, to assess whether predictors capture variance beyond both spatial position and 1/f spectral structure; and (iii) controlling for z, spectral exponent, and τ, the most stringent test. Incremental variance explained was assessed via OLS regression R².

## Spatial autocorrelation-corrected inference (spin tests)

Parametric p-values from partial correlations may be inflated by residual spatial autocorrelation. We therefore assessed the significance of all key findings using spin permutation tests (Alexander-Bloch et al., 2018). For each of 5,000 permutations, independent random rotation matrices (uniform on SO(3)) were applied to parcel centroids on the conte69 surface, separately for each hemisphere. Rotated centroids were matched to their nearest original parcel via Euclidean distance, and the predictor values were reassigned accordingly, preserving the spatial autocorrelation structure of the original map while disrupting its relationship with ρ. The partial correlation was then recomputed for each permuted map. Two-tailed p_spin values were calculated as the proportion of null partial correlations with absolute value ≥ the observed value, with a continuity correction of +1/(n_perms + 1).

## Data and code availability

| Resource | Source |
|----------|--------|
| Valk structural covariance gradients | https://github.com/sofievalk/projects |
| HCP group-average FC | BrainSpace `load_group_fc('schaefer', scale=400)` |
| Allen Human Brain Atlas | abagen toolbox (Markello et al., 2021) |
| Mesulam classification | BrainSpace `mesulam_conte69.csv` |
| BigBrain intensity profiles & MPC | https://github.com/MICA-MNI/micaopen (Paquola et al., 2019) |
| Glasser 360 conte69 labels | ENIGMA Toolbox (Larivière et al., 2021) |
| Schaefer 400 atlas | Schaefer et al., 2018 |

Analysis scripts are available at [GitHub repository URL].

## References

Amunts, K., et al. (2013). BigBrain: An ultrahigh-resolution 3D human brain model. *Science,* 340, 1472–1475.

Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage,* 178, 540–551.

Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature,* 536, 171–178.

Hawrylycz, M. J., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. *Nature,* 489, 391–399.

Margulies, D. S., et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS,* 113, 12574–12579.

Markello, R. D., et al. (2021). Standardizing workflows in imaging transcriptomics with the abagen toolbox. *eLife,* 10, e72129.

Pandya, D. N., et al. (2015). *Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept.* Oxford University Press.

Sanides, F. (1962). *Die Architektonik des menschlichen Stirnhirns.* Springer.

Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex,* 28, 3095–3114.

Valk, S. L., et al. (2020). Shaping brain structure: Genetic and phylogenetic axes of macroscale organization of cortical thickness. *Science Advances,* 6, eabb3417.

Vos de Wael, R., et al. (2020). BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. *Communications Biology,* 3, 103.

Larivière, S., et al. (2021). The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. *Nature Methods,* 18, 698–700.

Paquola, C., et al. (2019). Microstructural and functional gradients are increasingly dissociated in transmodal cortices. *PLOS Biology,* 17, e3000284.
