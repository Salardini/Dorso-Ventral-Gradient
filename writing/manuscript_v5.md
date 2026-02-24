# A dorsoventral gradient of rotational dynamics in human cortex

Arash Salardini1,2
1Glenn Biggs Institute for Alzheimer’s & Neurodegenerative Diseases, UT Health San Antonio, San Antonio, TX, USA. 2Department of Neurology, UT Health San Antonio, San Antonio, TX, USA.
Correspondence: salardini@uthscsa.edu


## Abstract

The cerebral cortex is organized along multiple spatial gradients, yet how local dynamical properties vary across the cortical surface remains unknown. Here, using resting-state magnetoencephalography (MEG) in 212 adults, we report a dorsoventral gradient of rotational dynamics—a linearized signature of oscillatory neural activity—quantified by an index ρ derived from delay-embedded autoregressive models. Ventral cortex exhibited stronger rotational dynamics than dorsal cortex (r = −0.72; Pspin < 0.00001), a pattern that replicated across cognitive states, parcellation schemes and imaging modalities. Systematic evaluation of candidate architectural predictors identified the dual-origin structural covariance gradient—reflecting phylogenetically conserved, genetically determined cortical connectivity along paleocortical and archicortical lineages—as the principal predictor of ρ, independent of spatial position, spectral composition and intrinsic timescale. These findings establish rotational dynamics as a novel axis of macroscale cortical organization rooted in developmental architecture.


## Introduction

The cerebral cortex exhibits systematic spatial organization across multiple dimensions. Functional connectivity gradients reveal a principal axis spanning unimodal to transmodal cortex1, intrinsic neural timescales lengthen along a similar hierarchy2,3, and cytoarchitectural differentiation grades from granular sensory to agranular limbic cortex4,5. These organizational principles constrain how cortical circuits process, integrate and transmit information. Yet one fundamental dimension has received little attention: the spatial distribution of local dynamical regimes.

Neural circuits can operate in qualitatively different dynamical regimes6. In some, perturbations decay monotonically toward equilibrium; in others, trajectories rotate through state space before settling—the hallmark of underdamped, oscillatory dynamics. These regimes carry distinct computational implications: decaying dynamics support point-attractor computations and evidence accumulation, while rotational dynamics enable sequence generation, rhythmic processing and temporal coding7,8. Whether cortical regions differ systematically in their local dynamical regime, and if so what determines this variation, are open questions with implications for understanding the computational architecture of the cortex.

Here we introduce ρ (rho), an index that quantifies the rotational component of local cortical dynamics from MEG recordings. The metric is derived from delay-embedded autoregressive models applied to parcel-level source-reconstructed time series: high ρ indicates underdamped, oscillatory dynamics, while low ρ indicates overdamped, decaying dynamics (Fig. 1a). This approach is distinct from population-level rotational dynamics identified by jPCA in motor cortex8; our metric characterizes the geometry of single-parcel trajectories in reconstructed state space following Takens’ embedding theorem9. Using resting-state MEG from 212 healthy adults, we discovered that ρ exhibits a pronounced dorsoventral gradient, with ventral regions showing stronger rotational dynamics. This gradient is distinct from established cortical axes, replicated across cognitive states and imaging modalities, and survived extensive confound controls.

We then asked what architectural feature of cortex determines this spatial distribution. The dual-origin model of cortical development holds that the cortical mantle differentiates from two phylogenetically ancient origins: paleocortex (piriform and olfactory cortex), giving rise to ventral and lateral neocortical regions, and archicortex (hippocampal formation), giving rise to dorsal and medial regions5,10,11. These lineages produce different connectivity topologies—locally concentrated connections in paleocortical derivatives versus long-range hierarchical circuits in archicortical derivatives—that are genetically specified and phylogenetically conserved12. Through systematic evaluation of candidate architectural predictors—including cytoarchitectural classification, laminar microstructure, interneuron gene expression, functional and structural connectivity and developmental gradients—we identified the dual-origin structural covariance gradient as the principal predictor of the ρ gradient, linking the macroscale organization of cortical dynamics to genetically determined developmental architecture.


## Results

### A dorsoventral gradient of rotational dynamics

We computed ρ for each of 400 cortical parcels13 from resting-state MEG in 212 participants (208 contributing to each parcel after quality control). The ρ metric quantifies the rotational component of local dynamics by fitting a vector autoregressive model to delay-embedded time series and computing the mean sine of eigenvalue angles (Methods). Across parcels, ρ exhibited a pronounced dorsoventral gradient (r = −0.72, 95% CI [−0.76, −0.68]; Pspin < 0.00001 based on 100,000 spin permutations; Fig. 1a–d). Ventral regions—including inferior temporal, orbitofrontal and ventral occipital cortex—showed high ρ values, while dorsal regions—including superior parietal, dorsolateral prefrontal and dorsal premotor cortex—showed low ρ values.

The MNI z-coordinate (dorsal–ventral axis) showed the strongest correlation with ρ (r = −0.72), compared with x (medial–lateral; r = −0.05) and y (anterior–posterior; r = 0.28). The gradient axis was oriented 18° from the pure dorsoventral direction (Fig. 1d), consistent across hemispheres and cortical divisions (Extended Data Table 1).

### The ρ gradient is robust across individuals, cognitive states and imaging modalities

The gradient persisted across parcellation resolutions (Schaefer-200: r = −0.70; Schaefer-100: r = −0.69; all Pspin < 0.0001) and was consistent across individuals: the median within-subject correlation was r = −0.33 (95% CI [−0.35, −0.30]), with 94% of participants showing a negative correlation (Wilcoxon W = 488, P < 10−33; Fig. 1e; Fig. 2c).

The dorsoventral gradient replicated during visual object recognition (N = 69; r = −0.68) and auditory speech processing (N = 95; r = −0.74), with gradient directions nearly identical to rest, indicating stable underlying circuit architecture (Fig. 2a). To test cross-modal generalizability, we computed ρ from resting-state fMRI in 68 participants. Despite dramatically different temporal resolution (TR = 2 s versus MEG’s 5-ms sampling), fMRI-derived ρ exhibited a significant dorsoventral gradient in the same direction (r = −0.35; Pspin = 0.0001; Fig. 1h). Additional fMRI-derived measures—intrinsic timescales, spectral exponent14 and fractional amplitude of low-frequency fluctuations15—all showed significant dorsoventral organization in directions consistent with MEG findings (Fig. 2e).

Because ρ correlates with the spectral exponent (r = −0.89), we tested whether the gradient reflects spectral composition rather than dynamical organization. Partial correlations controlling for the spectral exponent preserved the gradient (r = −0.59, Pspin = 0.0001), as did controls for total power (r = −0.68), gamma/delta ratio (r = −0.75) and all spectral confounds simultaneously (r = −0.27, Pspin = 0.0001). Controlling for source depth yielded r = −0.71 (Pspin = 0.0001). Permutation entropy16—a model-free nonlinear measure—showed the same dorsoventral gradient (r = −0.57; Pspin = 0.003) and correlated positively with ρ (r = +0.49), confirming that the gradient reflects genuine dynamical structure beyond spectral features.

### Frequency-specific organization and relationship with intrinsic timescales

Band-specific analysis revealed a striking frequency-dependent reversal (Fig. 1f; Extended Data Table 1). Slow rhythms showed dorsal predominance: delta (1–4 Hz), r = +0.55; theta (4–8 Hz), r = +0.47. Fast rhythms showed ventral predominance: alpha (8–13 Hz), r = −0.65; beta (13–30 Hz), r = −0.77; gamma (30–40 Hz), r = −0.73. All bands survived FDR correction (q < 0.001). The finding that motor and premotor cortices show low beta-band ρ despite being canonical beta generators17 may reflect the distinction between oscillatory power and dynamical regime: motor beta represents a stable, quasi-periodic rhythm—an attractor state with low trajectory curvature—consistent with its proposed role in maintaining the postural status quo18.

Intrinsic timescales (τ)2 showed no spatial organization separable from spatial autocorrelation (Pspin = 0.23)19, and in raw parcel space τ and ρ were independent (r = −0.03). However, after regressing out spatial coordinates from both maps, τ and ρ residuals showed strong anticorrelation (r = −0.62; Pspin = 0.0001; Fig. 1g)—suggesting that, within any cortical neighbourhood, circuits cannot simultaneously maximize temporal integration (long τ) and rotational dynamics (high ρ). This anticorrelation held within six of seven Yeo functional networks20 (Fisher z-weighted meta-analytic average r = −0.19, z = −3.71, P = 0.0002), confirming a genuine local circuit trade-off independent of macroscale spatial geometry.

Network-level mean ρ showed minimal variation across the sensory–transmodal axis (<1% range), while the dorsoventral gradient was present within every Yeo network20 (all |r| > 0.44, all P < 0.05). Controlling for network membership did not attenuate the dorsoventral correlation (partial r = −0.66). The ρ map was uncorrelated with myelin content21 (r = −0.015; Pspin = 0.80), confirming independence from established cortical gradients.

### The dual-origin developmental gradient predicts ρ

We systematically evaluated candidate architectural predictors using a hierarchical statistical framework: zero-order Spearman correlations, partial correlations controlling for the z-coordinate, and partial correlations controlling for z, spectral exponent and intrinsic timescale simultaneously (Fig. 3a). Spin permutation tests assessed robustness to spatial autocorrelation at every level. The z-coordinate alone explained 52.3% of ρ variance, and the spectral exponent alone explained 79.5%. Together they explained 86.8%.

Against this demanding baseline, the second structural covariance gradient (G2) from Valk et al.12—capturing the dual-origin developmental axis of cortical organization5,10,11—was the only predictor to survive all levels of statistical control (Fig. 3b). G2 is orthogonal to the MNI z-coordinate (ρs = 0.012) and captures a dimension of cortical organization entirely distinct from dorsoventral spatial position. Negative G2 values correspond to the paleocortical pole (regions developmentally closer to piriform cortex) and positive values to the archicortical pole (closer to hippocampal formation); the negative G2–ρ correlation indicates that paleocortical-lineage regions exhibit stronger rotational dynamics.

The G2–ρ relationship strengthened progressively as confounds were added: from ρs = −0.155 (zero-order) to r = −0.213 (controlling for z) to r = −0.249 (controlling for z, spectral exponent and τ; P = 4.7 × 10−7; Pspin = 0.018; Fig. 3c,d). This suppression pattern—where a predictor becomes stronger as covariates are added—indicates that the dominant spatial and spectral gradients had been partially masking the G2 signal. The genetic correlation gradient from Valk et al.12 replicated each result (r|z+SE+τ = −0.231, P = 3.0 × 10−6), confirming that the structural determinant of ρ is genetically specified. No other structural covariance gradient showed significant partial correlations with ρ (Fig. 3e).

### Alternative architectural predictors and structural connectivity

The systematic evaluation yielded a clear pattern of null results for intuitive candidate mechanisms (Fig. 3a). Interneuron gene expression from the Allen Human Brain Atlas22 (PVALB: ρs = −0.09, P = 0.11; SST: ρs = +0.12, P = 0.02; both null after z-correction) (Fig. 4c–f) did not predict ρ. No BigBrain cytoarchitectural measure23 showed a significant association after controlling for spatial position, including MPC Gradient 124 (ρs = −0.034), profile standard deviation (ρs = +0.067) and the Mesulam cytoarchitectural hierarchy (ρs = −0.027). Functional connectivity metrics were largely uninformative before spectral exponent control, but FC Gradient 11 and FC clustering coefficient both emerged as significant after accounting for the spectral exponent (r|z+SE+τ = −0.235 and +0.188, respectively).

The G2–ρ relationship held within cytoarchitectural classes, with the strongest effect in paralimbic cortex (r = −0.34, P = 0.002, PFDR = 0.014)—precisely where paleocortical and archicortical derivatives intermingle (Fig. 3f). Heteromodal (r = −0.20) and unimodal cortex (r = −0.18) showed consistent trends, confirming that the dual-origin gradient captures organizational variation within cytoarchitectural classes, not merely the distinction between them.

Structural connectivity data from the Human Connectome Project25 provided direct empirical support (Fig. 3g). The short/long range ratio was the strongest zero-order structural connectivity predictor (ρs = −0.273, P < 0.001) and survived the full control battery (r|z+SE+τ = −0.115, P < 0.05). G2 correlated with SC connectivity range (G2 versus short/long ratio: ρs = +0.147, P = 0.003), confirming that paleocortical-pole regions have more locally concentrated structural connections. Mediation analysis revealed that G2 fully survived controlling for the SC short/long ratio (r = −0.238, P = 1.4 × 10−6), while the SC ratio was reduced to non-significance after controlling for G2 (r = −0.089, P = 0.074), indicating that G2 captures the connectivity architecture that SC indexes, plus additional developmental information that tractography alone does not reveal (Fig. 3h).


## Discussion

We have identified a dorsoventral gradient of rotational dynamics in human cortex: ventral regions exhibit stronger rotational dynamics, while dorsal regions exhibit more integrative dynamics with longer intrinsic timescales. This gradient is robust across parcellation schemes, cognitive states and imaging modalities, and is not explained by spectral composition, source depth, network membership or myelin content. Systematic evaluation of candidate architectural predictors identified the dual-origin structural covariance gradient12—reflecting the phylogenetically conserved, genetically determined organization of cortical connectivity along paleocortical and archicortical lineages—as the only predictor that explains ρ variance beyond the dominant spatial and spectral dimensions.

The null results are collectively informative. The ρ gradient does not track cytoarchitectural type, laminar differentiation23,24, the sensorimotor–transmodal hierarchy1 or regional variation in inhibitory interneuron subtypes22,26,27—despite the theoretical plausibility of each candidate. The failure of the interneuron hypothesis is particularly notable: macroscale PVALB expression gradients do not map onto macroscale dynamical gradients in the straightforward manner that local circuit models would predict28,29.

The link between dual-origin developmental architecture and rotational dynamics can be understood through connectivity topology. Paleocortical derivatives tend toward locally concentrated connectivity, creating conditions for tight excitatory–inhibitory loops that generate the antisymmetric coupling producing complex eigenvalues—the mathematical signature of rotation30. Recent computational work has shown that distance-dependent connectivity alone suffices to produce hierarchically modular networks with convergent population dynamics31. Archicortical derivatives, with longer-range hierarchical connections10, support feedforward integration, yielding lower ρ. The strong correlation between ρ and the spectral exponent (r = −0.90) is consistent: locally concentrated connectivity simultaneously steepens the spectral slope14 and generates rotational dynamics, because the same circuit architecture drives both.

The incremental variance explained by G2 beyond spatial position and spectral exponent is modest (0.9%), but this requires careful interpretation. The spectral exponent partially mediates the G2→ρ relationship: locally concentrated connectivity simultaneously steepens the spectral slope and generates rotational dynamics. The spectral exponent’s 79.5% R2 already captures the primary dynamical expression of the architectural gradient. The 0.9% increment represents the direct developmental effect on ρ independent of its spectral consequences. The appropriate metric is the partial correlation (r = −0.249, P = 4.7 × 10−7, Pspin = 0.018), which is robust to spatial autocorrelation and replicated with the genetically determined gradient.

Several limitations warrant consideration. The structural covariance gradient is an indirect measure of connectivity architecture; direct assessment using diffusion tractography would provide stronger evidence. The AHBA gene expression data derive from only six donors with limited bilateral coverage22. The fMRI-derived ρ did not correlate with MEG-derived ρ at the parcel level, consistent with these modalities capturing different temporal scales but limiting direct cross-modal validation. Although we obtained vertex-level cross-parcellation mapping for the BigBrain analysis, the resolution mismatch between 360 Glasser32 and 400 Schaefer13 parcels means that some spatial precision is lost.

The dual-origin framework generates testable predictions. Because the structural covariance gradient is phylogenetically conserved12, the ρ gradient should be present in non-human primates. NMDA receptor antagonists should selectively reduce ρ in high-recurrence ventral regions. Neurodevelopmental conditions with disrupted local-versus-long-range connectivity balance may show altered ρ gradients, and in neurodegenerative disease the known early vulnerability of paleocortical-lineage regions raises the possibility that ρ is differentially affected where pathology first accumulates. Laminar-resolved MEG33 could examine whether rotational dynamics originate preferentially in supragranular layers where local recurrent connections are densest.


## Methods

Participants. We analysed resting-state MEG from the Mother of Unification Studies (MOUS) dataset34. After quality control, 212 participants were included (ages 18–45, mean 26.3 years), with 208 contributing to each parcel. Task MEG was available for subsets: visual (N = 69), auditory (N = 95). Resting-state fMRI was available for 68 participants.

MEG acquisition and preprocessing. MEG was recorded using a 275-channel CTF system at 1,200 Hz. Preprocessing comprised band-pass filtering (1–40 Hz), notch filtering (50 Hz), independent component analysis artefact removal and downsampling to 200 Hz.

Source reconstruction. Source localization used LCMV beamforming35 with single-shell boundary element models. Source time series were extracted for 400 cortical parcels13.

Rotational dynamics index (ρ). For each parcel time series x(t), we constructed a ten-dimensional state vector: X(t) = [x(t), x(t − d), …, x(t − 9d)] where d = 1 sample (5 ms), following Takens’ embedding theorem9. A VAR(1) model X(t + 1) = A·X(t) + ε was fit with ridge regularization (α = 0.001). The rotational index ρ = mean(|Im(λ)| / |λ|) for eigenvalues λ with |λ| > 0.01, equalling the mean sine of eigenvalue angles. Model fit R2 (median = 0.9999) confirmed adequate approximation.

fMRI acquisition and preprocessing. Resting-state fMRI was acquired with TR = 2 s (266 volumes per subject). Preprocessing used nilearn36: standardization, detrending and bandpass filtering (0.01–0.1 Hz). fMRI-derived ρ used five-dimensional delay embedding with delay = 1 TR. Intrinsic timescale τ was computed as the integral of the autocorrelation function2. Spectral exponent was the slope of the log–log power spectrum. fALFF was computed following Zou et al.15.

Spatial statistics. Spin permutation tests19 (100,000 rotations for primary MEG analyses; 10,000 for fMRI; 5,000 for architectural predictor analyses) preserved hemisphere structure.

Architectural predictor analysis. For each candidate predictor, we computed: (1) zero-order Spearman correlation with ρ; (2) partial correlation controlling for the z-coordinate; (3) partial correlation controlling for z and spectral exponent; (4) partial correlation controlling for z, spectral exponent and τ. Variance partitioning used sequential R2 from OLS regression. Structural covariance gradients were from Valk et al.12. BigBrain data were from Amunts et al.23 and Paquola et al.24; cross-parcellation mapping used vertex-level Glasser32 labels via the ENIGMA Toolbox25. HCP structural connectivity used log-transformed streamline counts25. Gene expression used AHBA data22 parcellated via abagen.

Full details on frequency-specific analysis, cortical flatmap visualization, mediation analysis and all additional methods are provided in the Supplementary Methods.


## Data availability

The MOUS dataset is available from the Donders Repository34. HCP data are available from the Human Connectome Project via the ENIGMA Toolbox25. BigBrain data are available from the MICA laboratory. AHBA data are available from the Allen Institute22.


## Code availability

Analysis code is available at https://github.com/Salardini/Dorso-Ventral-Gradient.


## Acknowledgements

[To be completed]


## Author contributions

A.S. conceptualized the study, developed the methodology, performed all analyses, created all visualizations and wrote the manuscript.


## Competing interests

The author declares no competing interests.


## Figure legends

**Fig. 1 | A dorsoventral gradient of rotational dynamics in human cortex.**

a, Schematic of the ρ metric. Delay-embedded time series are fit with a VAR(1) model; eigenvalues with large imaginary components indicate rotational (underdamped) dynamics, while purely real eigenvalues indicate decaying (overdamped) dynamics. b, Cortical surface maps of group-average ρ (n = 212 participants, Schaefer 400 parcellation). Left and right hemispheres shown in lateral and medial views. c, Cortical flatmap generated by multidimensional scaling of parcel centroids, with Voronoi tessellation. Hemispheres shown separately; dorsal–ventral gradient is clearly visible in both. d, Scatter plot of ρ versus z-coordinate (dorsoventral axis) for all 400 parcels, coloured by Yeo 7-network assignment. Black curve: quadratic fit. Spearman ρₛ = −0.73, p_spin < 10⁻⁵. e, Distribution of individual-level Spearman correlations between ρ and z across 25 participants. Mean r = −0.43; 100% of participants showed negative correlations. f, Band-specific ρ–z correlations. Slow rhythms (δ, θ) show dorsal predominance (positive r); fast rhythms (α, β, γ) show ventral predominance (negative r). All p_spin < 0.001. g, Scatter plot of ρ versus τ (intrinsic timescale) after residualizing both for spatial coordinates (x, y, z). Spearman ρₛ = −0.34, confirming a local circuit trade-off independent of macroscale position. h, fMRI-derived ρ versus z-coordinate (n = 68 participants). Spearman ρₛ = 0.13, indicating a weak but directionally consistent gradient in the BOLD signal.

**Fig. 2 | Robustness and cross-modal replication.**

a, Task replication. ρ versus z during auditory speech processing (n = 95; r = −0.74) and visual object recognition (n = 69; r = −0.68) overlaid on resting state (grey; r = −0.73). Gradient directions are nearly identical across cognitive states. b, HCP fMRI replication (n = 1,096 participants). fMRI-derived ρ shows no significant dorsoventral gradient (ρₛ ≈ 0.00, p = 0.99), consistent with the mechanistic prediction that rotational dynamics operate at fast timescales inaccessible to BOLD imaging. c, Atlas sensitivity. The ρ–z gradient is stable across Schaefer parcellation resolutions: 400 parcels (r = −0.72), 200 parcels (r = −0.70), 100 parcels (r = −0.69). All p_spin < 0.001. d, Adaptive versus fixed embedding delay. Band-specific ρ–z correlations computed with delays matched to each frequency band’s timescale (red) compared to fixed 5 ms delay (blue). The gradient is robust to delay choice. e, fMRI-derived complementary measures versus z-coordinate. ρ_fMRI (r = 0.13), τ_fMRI (r = −0.17), and spectral exponent (r = 0.14) all show significant dorsoventral organization. f, Summary of ρ–z correlations across all datasets and modalities.

**Fig. 3 | The dual-origin structural covariance gradient predicts rotational dynamics.**

a, Heatmap of partial Spearman correlations between 12 candidate architectural predictors and ρ under four levels of statistical control: zero-order, controlling for z, controlling for z + spectral exponent (SE), and controlling for z + SE + intrinsic timescale (τ). Significance markers: *p < 0.05, **p < 0.01, ***p < 0.001. b, Cortical surface maps of SCov G2 (Valk et al., 2020). Left and right hemispheres shown in lateral and medial views. c, ρ versus SCov G2 for all 400 parcels, coloured by network. Zero-order ρₛ = −0.155; partial r|z,SE,τ = −0.249, p = 4.7 × 10⁻⁷. d, Progressive strengthening of G2–ρ association as confounds are added, from zero-order (−0.155) through full controls (−0.249). e, All candidate predictors ranked by |partial r| under the most stringent control (z + SE + τ). Category colours indicate predictor class. f, ρ versus SCov G2 within each Mesulam cytoarchitectural class, with per-class regression lines. The G2–ρ relationship holds within cortical hierarchy levels (overall partial r|z,SE,τ,Mesulam = −0.256, p = 2.1 × 10⁻⁷). g, ρ versus SC short/long range ratio (ρₛ = −0.273; r|z,SE,τ = −0.115). h, Mediation analysis. G2 survives controlling for SC metrics (r = −0.238), while SC short/long ratio is reduced to non-significance after controlling for G2 (r = −0.089, n.s.).

**Fig. 4 | Excitatory–inhibitory balance model and interneuron gene expression.**

a, E–I network model. Varying inhibitory gain (g_I) produces opposing effects on timescale τ (blue) and rotational index ρ (red), demonstrating that a single circuit parameter can generate the observed τ–ρ trade-off. b, Model-derived τ versus ρ across inhibitory gain values, coloured by g_I. The model produces a strong anticorrelation (r = −0.84, p = 0.0006), qualitatively matching the empirical τ–ρ trade-off. c, Spearman correlations between interneuron marker gene expression (Allen Human Brain Atlas) and ρ. SST shows a weak positive association (r = 0.12, p = 0.02) and PV−SST a weak negative association (r = −0.12, p = 0.02); neither survives spatial confound control. d–f, Scatter plots of ρ versus PVALB expression (d; r = −0.09, n.s.), SST expression (e; r = 0.12, p = 0.02), and PV−SST ratio (f; r = −0.12, p = 0.02).


## References

1. Margulies, D. S. et al. Situating the default-mode network along a principal gradient of macroscale cortical organization. Proc. Natl Acad. Sci. USA 113, 12574–12579 (2016).

2. Honey, C. J. et al. Slow cortical dynamics and the accumulation of information over long timescales. Neuron 76, 423–434 (2012).

3. Murray, J. D. et al. A hierarchy of intrinsic timescales across primate cortex. Nat. Neurosci. 17, 1661–1663 (2014).

4. Felleman, D. J. & Van Essen, D. C. Distributed hierarchical processing in the primate cerebral cortex. Cereb. Cortex 1, 1–47 (1991).

5. Sanides, F. Die Architektonik des menschlichen Stirnhirns (Springer, 1962).

6. Strogatz, S. H. Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering (Westview Press, 2015).

7. Hennequin, G., Vogels, T. P. & Gerstner, W. Optimal control of transient dynamics in balanced networks supports generation of complex movements. Neuron 82, 1394–1406 (2014).

8. Churchland, M. M. et al. Neural population dynamics during reaching. Nature 487, 51–56 (2012).

9. Takens, F. Detecting strange attractors in turbulence. Lect. Notes Math. 898, 366–381 (1981).

10. Pandya, D. N., Seltzer, B., Petrides, M. & Cipolloni, P. B. Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept (Oxford Univ. Press, 2015).

11. Sanides, F. Functional architecture of motor and sensory cortices in primates in the light of a new concept of neocortex evolution. in The Primate Brain (eds Noback, C. R. & Montagna, W.) 137–208 (Appleton-Century-Crofts, 1970).

12. Valk, S. L. et al. Shaping brain structure: genetic and phylogenetic axes of macroscale organization of cortical thickness. Sci. Adv. 6, eabb3417 (2020).

13. Schaefer, A. et al. Local–global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cereb. Cortex 28, 3095–3114 (2018).

14. Gao, R., Peterson, E. J. & Voytek, B. Inferring synaptic excitation/inhibition balance from field potentials. NeuroImage 158, 70–78 (2017).

15. Zou, Q. H. et al. An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF. J. Neurosci. Methods 172, 137–141 (2008).

16. Bandt, C. & Pompe, B. Permutation entropy: a natural complexity measure for time series. Phys. Rev. Lett. 88, 174102 (2002).

17. Pfurtscheller, G. & Lopes da Silva, F. H. Event-related EEG/MEG synchronization and desynchronization: basic principles. Clin. Neurophysiol. 110, 1842–1857 (1999).

18. Engel, A. K. & Fries, P. Beta-band oscillations—signalling the status quo? Curr. Opin. Neurobiol. 20, 156–165 (2010).

19. Alexander-Bloch, A. F. et al. On testing for spatial correspondence between maps of human brain structure and function. NeuroImage 178, 540–551 (2018).

20. Yeo, B. T. T. et al. The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J. Neurophysiol. 106, 1125–1165 (2011).

21. Glasser, M. F. & Van Essen, D. C. Mapping human cortical areas in vivo based on myelin content as revealed by T1- and T2-weighted MRI. J. Neurosci. 31, 11597–11616 (2011).

22. Hawrylycz, M. J. et al. An anatomically comprehensive atlas of the adult human brain transcriptome. Nature 489, 391–399 (2012).

23. Amunts, K. et al. BigBrain: an ultrahigh-resolution 3D human brain model. Science 340, 1472–1475 (2013).

24. Paquola, C. et al. Microstructural and functional gradients are increasingly dissociated in transmodal cortices. PLoS Biol. 17, e3000284 (2019).

25. Larivière, S. et al. The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. Nat. Methods 18, 698–700 (2021).

26. Rudy, B. et al. Three groups of interneurons account for nearly 100% of neocortical GABAergic neurons. Dev. Neurobiol. 71, 45–61 (2011).

27. Sohal, V. S. et al. Parvalbumin neurons and gamma rhythms enhance cortical circuit performance. Nature 459, 698–702 (2009).

28. Cardin, J. A. et al. Driving fast-spiking cells induces gamma rhythm and controls sensory responses. Nature 459, 663–667 (2009).

29. Singer, W. & Gray, C. M. Visual feature integration and the temporal correlation hypothesis. Annu. Rev. Neurosci. 18, 555–586 (1995).

30. Murphy, B. K. & Miller, K. D. Balanced amplification: a new mechanism of selective amplification of neural activity patterns. Neuron 61, 635–648 (2009).

31. Guarino, D. et al. Convergent information flows in cortical networks reveal reproducible dynamics without attractor architecture. Nat. Neurosci. (2026).

32. Glasser, M. F. et al. A multi-modal parcellation of human cerebral cortex. Nature 536, 171–178 (2016).

33. Troebinger, L. et al. Discrimination of cortical laminae using MEG. NeuroImage 102, 885–893 (2014).

34. Schoffelen, J. M. et al. A 204-subject multimodal neuroimaging dataset to study language processing. Sci. Data 6, 17 (2019).

35. Van Veen, B. D. et al. Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Trans. Biomed. Eng. 44, 867–880 (1997).

36. Abraham, A. et al. Machine learning for neuroimaging with scikit-learn. Front. Neuroinformatics 8, 14 (2014).

37. Steriade, M., McCormick, D. A. & Sejnowski, T. J. Thalamocortical oscillations in the sleeping and aroused brain. Science 262, 679–685 (1993).

38. Goodale, M. A. & Milner, A. D. Separate visual pathways for perception and action. Trends Neurosci. 15, 20–25 (1992).

39. Onnela, J.-P. et al. Intensity and coherence of motifs in weighted complex networks. Phys. Rev. E 71, 065103 (2005).
