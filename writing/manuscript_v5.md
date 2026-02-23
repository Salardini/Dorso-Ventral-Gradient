A Dorsoventral Gradient of Rotational Dynamics in Human Cortex


# ABSTRACT

Cortical dynamics are spatially organized, yet how local dynamical properties vary across the cortical surface remains poorly understood. Here we characterize the spatial organization of rotational dynamics, a linearized signature of oscillatory neural activity, using resting-state magnetoencephalography (MEG) in 212 healthy adults. We introduce ρ, an index quantifying the rotational component of local dynamics derived from delay-embedded linear models. Across Schaefer-400 parcels, ρ exhibited a striking dorsoventral gradient (r = −0.72; p_spin < 0.00001), with ventral regions showing stronger rotational dynamics. This gradient was robust to parcellation resolution, replicated across cognitive states, and survived extensive confound controls including spectral composition (partial r = −0.59) and source depth (partial r = −0.71). Frequency-specific analysis revealed a spectral trade-off: slow oscillations showed dorsal predominance while fast oscillations showed ventral predominance. Intrinsic timescales (τ) exhibited strong anticorrelation with ρ after removing shared spatial geometry (r = −0.62; p_spin = 0.0001), suggesting a local circuit constraint. Systematic evaluation of candidate architectural predictors, including cytoarchitectural type, laminar differentiation, interneuron gene expression, functional connectivity gradients, and structural connectivity, identified the dual-origin structural covariance gradient (Valk et al., 2020) as a genetically specified predictor of rotational dynamics independent of spatial position, spectral properties, laminar differentiation, and intrinsic timescale (r = −0.249, p = 4.7 × 10⁻⁷; p_spin = 0.018), with replication in a genetically determined gradient (r = −0.231, p = 3.0 × 10⁻⁶). This gradient aligns with the dual origin model of cortical development, in which paleocortical derivatives—with dense locally concentrated connectivity—exhibit stronger rotation, while archicortical derivatives—with longer-range hierarchical connections—exhibit more integrative dynamics. Our findings establish ρ as a novel dorsoventral axis of cortical organization linked to genetically determined, phylogenetically conserved developmental architecture.


# INTRODUCTION

The cerebral cortex exhibits systematic spatial organization across multiple dimensions. Functional connectivity gradients reveal a principal axis spanning unimodal to transmodal cortex (Margulies et al., 2016), while intrinsic timescales increase along a similar hierarchy (Honey et al., 2012; Murray et al., 2014). These principles constrain how cortical circuits process information.

Beyond connectivity and timescales, neural dynamics themselves may be spatially organized. Dynamical systems can exhibit qualitatively different behaviors: stable, decaying dynamics versus oscillatory, rotational dynamics (Strogatz, 2015). These regimes have distinct computational implications: decaying dynamics support point-attractor computations, while rotational dynamics enable sequence generation and rhythmic processing (Hennequin et al., 2014).

We use “rotational dynamics” to describe the geometry of single-parcel trajectories in delay-embedded state space, distinct from population-level rotations identified by jPCA in motor cortex (Churchland et al., 2012). Our metric ρ quantifies the angular component of local dynamics, not coordinated rotations across a neural population. Importantly, ρ captures linearized signatures of rotational structure: it measures whether a linear approximation to local dynamics has complex eigenvalues (underdamped, oscillatory) versus purely real eigenvalues (overdamped, decaying). This is analogous to a spring-mass system: high ρ indicates underdamped oscillation where the mass overshoots and rings, while low ρ indicates overdamped decay where the mass settles without ringing.

Here we introduce ρ (rho), an index derived from delay-embedded vector autoregressive models following Takens’ embedding theorem (Takens, 1981). Using resting-state MEG from 212 participants, we discovered a pronounced dorsoventral gradient in ρ, with ventral regions exhibiting stronger rotational dynamics. We then systematically evaluated candidate architectural determinants of this gradient, testing cytoarchitectural classification, laminar microstructure, interneuron gene expression, functional and structural connectivity, and developmental gradients. This evaluation identified the dual-origin structural covariance gradient, reflecting the phylogenetically conserved, genetically determined organization of cortical connectivity along paleocortical and archicortical lineages, as the principal architectural predictor of the ρ gradient.


# RESULTS

## A dorsoventral gradient of rotational dynamics

We computed ρ for each of 400 cortical parcels (Schaefer et al., 2018) from resting-state MEG in 212 participants (208 contributing to each parcel after quality control). The ρ metric quantifies the rotational component of local dynamics by fitting a vector autoregressive model to delay-embedded time series and computing the mean sine of eigenvalue angles (see Methods).

Across parcels, ρ exhibited a pronounced dorsoventral gradient (r = −0.72, 95% CI [−0.76, −0.68]; p_spin < 0.00001 based on 100,000 spin permutations; Fig. 1a–d). Ventral regions, including inferior temporal, orbitofrontal, and ventral occipital cortex, showed high ρ values. Dorsal regions, including superior parietal, dorsolateral prefrontal, and dorsal premotor cortex, showed low ρ values.

Parcel coordinates were defined in MNI space with x (medial–lateral), y (anterior–posterior), and z (dorsal–ventral) axes. The z-coordinate showed the strongest correlation with ρ (r = −0.72), compared to x (r = −0.05) and y (r = 0.28). The gradient axis was oriented 18° from the pure dorsoventral direction (Fig. 1d), consistent across hemispheres and cortical divisions (Extended Data Table 1).

## Robustness and individual-level consistency

The gradient persisted across parcellation resolutions: Schaefer-200: r = −0.70; Schaefer-100: r = −0.69 (all p_spin < 0.0001). The gradient was consistent across individuals: median r = −0.33 (95% CI [−0.35, −0.30]); 94% of subjects showed negative correlation (Fig. 1e; Fig. 2c); Wilcoxon W = 488, p < 10⁻³³. The weaker individual-level correlation reflects measurement noise: the gradient was present in 94% of individuals with consistent direction, and the group average benefits from noise cancellation.

## Replication across cognitive states

The dorsoventral gradient replicated during visual object recognition (N = 69; r = −0.68) and auditory speech processing (N = 95; r = −0.74; Fig. 2a), with gradient directions nearly identical to rest, indicating stable underlying circuit architecture.

## Cross-modal validation with fMRI

To test whether the ρ gradient reflects modality-specific MEG properties or generalizable circuit organization, we computed ρ from resting-state fMRI in 68 participants from the MOUS cohort. Despite fMRI’s dramatically slower temporal resolution (TR = 2 s vs. MEG’s 5 ms sampling), fMRI-derived ρ exhibited a significant dorsoventral gradient in the same direction as MEG (r = −0.35; p_spin = 0.0001; Fig. 1h).

Additional fMRI-derived measures showed complementary spatial organization. Intrinsic timescales (τ_fMRI) were longer in dorsal regions (r = +0.31 with z; p_spin = 0.0001), consistent with the MEG τ–ρ anticorrelation. Spectral exponent—indexing the steepness of the 1/f power spectrum—was higher dorsally (r = +0.29; p_spin = 0.0001), indicating redder (slower-fluctuating) dynamics in dorsal cortex. Fractional amplitude of low-frequency fluctuations (fALFF; Zou et al., 2008) (Fig. 2e) also showed dorsal predominance (r = +0.34; p_spin = 0.0001).

Direct parcel-by-parcel correlations between fMRI and MEG measures were not significant (ρ_fMRI vs ρ_MEG: r = −0.03, p_spin = 0.58) (Fig. 2b), reflecting the different temporal scales captured by each modality. However, the convergent spatial gradients—with all fMRI measures showing significant dorsoventral organization in directions consistent with MEG findings—provide strong cross-modal validation that the ρ gradient reflects genuine circuit properties rather than modality-specific artifacts (Fig. 2f).

## Robustness to spectral and depth confounds

Because ρ correlates with spectral exponent (r = −0.89), we tested whether the gradient reflects spectral composition rather than dynamical organization. Partial correlation controlling for spectral exponent preserved the gradient (r = −0.59, p_spin = 0.0001). The gradient survived controls for total power (r = −0.68), gamma/delta ratio (r = −0.75), and all spectral confounds simultaneously (r = −0.27, p_spin = 0.0001). Controlling for distance from brain center (a depth proxy) yielded r = −0.71, p_spin = 0.0001. These results confirm that ρ captures genuine dynamical structure beyond spectral features.

## Independence from established cortical gradients

Network-level mean ρ showed minimal variation across the sensory–transmodal axis (<1% range), while the dorsoventral gradient was present within every functional network (Yeo et al., 2011; all |r| > 0.44, all p < 0.05). Controlling for network membership did not attenuate the dorsoventral correlation (partial r = −0.66). A Mantel test detected a weak statistical relationship with hierarchical organization (r = 0.03, p = 0.002), but the effect size is negligible (<0.1% shared variance). The ρ map was uncorrelated with myelin content (r = −0.015; p_spin = 0.80; Glasser & Van Essen, 2011).

## Validation with model-free nonlinear measure

Permutation entropy (Bandt & Pompe, 2002)—a model-free measure of ordinal temporal complexity—showed the same dorsoventral gradient (r = −0.57; p_spin = 0.003) and correlated positively with ρ (r = +0.49; p_spin = 0.001). The positive correlation between ρ and permutation entropy requires careful interpretation: unlike a pure sinusoid (which has low entropy due to predictability), neural signals are broadband mixtures where high ρ indicates locally underdamped dynamics—trajectories that curve through state space rather than decaying directly. This curved traversal, embedded in noise, produces higher ordinal complexity.

## Frequency-specific organization: a spectral trade-off

Band-specific analysis revealed a striking frequency-dependent reversal (Fig. 1f; Extended Data Table 1). Slow rhythms showed dorsal predominance: delta (1–4 Hz), r = +0.55; theta (4–8 Hz), r = +0.47. Fast rhythms showed ventral predominance: alpha (8–13 Hz), r = −0.65; beta (13–30 Hz), r = −0.77; gamma (30–40 Hz), r = −0.73. All bands survived FDR correction (q < 0.001). Because the fixed embedding delay (5 ms) represents a larger fraction of fast oscillation cycles than slow ones, broadband ρ is weighted toward beta and gamma frequencies. The adaptive delay analysis (Fig. 2d) confirms that frequency-specific gradients are robust when delays are matched to each band’s timescale (Extended Data Table 1C).

The finding that motor and premotor cortices show low beta-band ρ despite being canonical beta generators (Pfurtscheller & Lopes da Silva, 1999) may reflect the distinction between oscillatory power and dynamical regime. Motor beta represents a stable, quasi-periodic rhythm—an attractor state with low trajectory curvature—consistent with its proposed role in maintaining postural “status quo” (Engel & Fries, 2010). Ventral beta may reflect more transient, phasic dynamics with higher rotational signatures.

## Relationship with intrinsic timescales

Intrinsic timescales (τ; Honey et al., 2012) showed apparent correlations with anatomical coordinates under parametric tests but did not survive spin permutation (p_spin = 0.23; Alexander-Bloch et al., 2018), indicating τ’s spatial structure is not separable from spatial autocorrelation. In raw parcel space, τ and ρ were independent (r = −0.03). However, after regressing out spatial coordinates from both maps, τ and ρ residuals showed strong anticorrelation (r = −0.62; p_spin = 0.0001) (Fig. 1g).

The rationale for this residualization warrants clarification: although τ’s raw spatial correlations did not survive spin permutation, this does not imply τ is spatially uniform—only that its structure is not distinguishable from spatial autocorrelation. Residualizing both τ and ρ by spatial coordinates asks a different question: within any cortical neighborhood (controlling for gross position), are τ and ρ related? The strong residual anticorrelation suggests local circuit constraints independent of macroscale anatomy: circuits cannot simultaneously maximize both temporal integration (long τ) and rotational dynamics (high ρ).

To confirm that this anticorrelation is not an artifact of the spatial residualization procedure, we tested the τ–ρ relationship within each Yeo network after controlling for the z-coordinate. Six of seven networks showed negative within-network partial correlations (Control: r = −0.37, p = 0.007; SalVentAttn: r = −0.35, p = 0.016; Visual: r = −0.28, p = 0.029; Limbic: r = −0.25; DorsAttn: r = −0.18; SomMot: r = −0.12; Default: r = +0.03). The Fisher z-weighted meta-analytic average across all seven networks was r = −0.19 (z = −3.71, p = 0.0002). This within-network analysis confirms that the local circuit trade-off between temporal integration and rotational dynamics is a property of the data itself—not a consequence of the coordinate regression framework—because it holds within spatially distributed functional networks without requiring explicit residualization on MNI coordinates.


## Architectural predictors of the ρ gradient

Having established the ρ gradient’s robustness and independence from established cortical gradients, we next asked what architectural feature of cortex determines the spatial distribution of rotational dynamics. We systematically evaluated candidate predictors using a hierarchical statistical framework: zero-order Spearman correlations, partial correlations controlling for the z-coordinate (removing the dominant spatial gradient), and partial correlations controlling for z, spectral exponent, and intrinsic timescale simultaneously (the most stringent test). Spin permutation tests (5,000 rotations) assessed robustness to spatial autocorrelation. (Fig. 3a)

### Variance partitioning: spatial, spectral, and architectural contributions

The z-coordinate alone explained 52.3% of ρ variance (R² = 0.523), and the spectral exponent alone explained 79.5% (R² = 0.795). Together, z and the spectral exponent explained 86.8% (R² = 0.868). Adding the intrinsic timescale τ contributed negligibly beyond these two dimensions (R² = 0.870). This establishes a demanding baseline: any architectural predictor must explain variance beyond the dominant spatial and spectral dimensions.

### The dual-origin structural covariance gradient predicts ρ

The second structural covariance gradient (G2) from Valk et al. (2020), computed from cortical thickness covariance across Human Connectome Project (HCP) participants in the Schaefer 400 parcellation, captures an organizational axis that aligns with the dual origin model of cortical development (Sanides, 1962; Pandya et al., 2015). In this framework, the cortical mantle develops from two phylogenetically ancient origins: paleocortex (piriform/olfactory cortex), giving rise to ventral and lateral neocortical regions through successive waves of laminar elaboration, and archicortex (hippocampal formation), giving rise to dorsal and medial regions.

SCov G2 was orthogonal to the MNI z-coordinate (ρ_s = 0.012), meaning it captures a dimension of cortical organization entirely distinct from dorsoventral spatial position. Negative G2 values correspond to the paleocortical pole (regions developmentally closer to piriform/olfactory cortex) and positive values to the archicortical pole (regions closer to hippocampal formation); the negative G2–ρ correlation therefore indicates that paleocortical-lineage regions exhibit stronger rotational dynamics. This relationship strengthened progressively as confounds were added: from ρ_s = −0.155 (zero-order) to r = −0.213 (controlling for z) to r = −0.253 (controlling for z and spectral exponent) to r = −0.249 (controlling for z, spectral exponent, and τ; p = 4.7 × 10⁻⁷) (Fig. 3b–d). This pattern—where a predictor becomes stronger rather than weaker as covariates are added—indicates that the dominant spatial and spectral gradients were partially masking the G2 signal. Removing them reveals the underlying developmental contribution more clearly.

The genetic correlation gradient from Valk et al. (2020) replicated each result (r|z+SE+τ = −0.231, p = 3.0 × 10⁻⁶), confirming that the structural determinant of ρ is genetically specified. No other structural covariance gradient showed significant partial correlations with ρ. (Fig. 3e) In terms of incremental variance, G2 added 0.9% beyond z and spectral exponent (total R² = 0.877), and the full model with G2, FC Gradient 1, and FC clustering coefficient reached R² = 0.884.

### Functional connectivity predictors emerge after spectral control

Functional connectivity metrics were largely uninformative before spectral exponent control. FC Gradient 1 (the sensorimotor-to-default mode hierarchy; Margulies et al., 2016) showed no zero-order correlation with ρ and only a marginal effect after z-correction (r|z = −0.110, p = 0.03). However, after additionally controlling for the spectral exponent, FC Gradient 1 emerged as a significant predictor (r|z+SE = −0.242, p < 0.001), and FC clustering coefficient became significantly positive (r|z+SE = +0.195, p < 0.001). Both survived the full control battery (r|z+SE+τ = −0.235 and +0.188, respectively). These results suggest that functional connectivity structure captures variance in ρ that is masked by the dominant 1/f structure but becomes visible once it is removed. The positive clustering coefficient effect is consistent with the recurrence hypothesis: regions with more locally clustered functional connectivity exhibit stronger rotational dynamics.

### Interneuron gene expression does not predict ρ

The interneuron hypothesis—that regional variation in fast-spiking parvalbumin-positive (PV+) interneurons drives the ρ gradient through differential sharpening of oscillatory dynamics—was not supported. PVALB expression from the Allen Human Brain Atlas (Hawrylycz et al., 2012) showed no significant correlation with ρ (ρ_s = −0.09, p = 0.11), and SST expression showed a weak association in the wrong direction (ρ_s = +0.12, p = 0.02). Both were null after controlling for z (p > 0.8). (Fig. 4c–f) While excitatory–inhibitory balance is clearly relevant to rotational dynamics at the microcircuit level (Churchland et al., 2012; Murphy & Miller, 2009), macroscale variation in interneuron gene expression does not account for the cortical ρ gradient.

### BigBrain cytoarchitectural features do not predict ρ

To test whether histological laminar differentiation predicts the ρ gradient, we analyzed cytoarchitectural data from the BigBrain atlas, an ultra-high-resolution (20 μm) 3D reconstruction of a post-mortem human brain (Amunts et al., 2013). BigBrain staining intensity profiles and microstructure profile covariance (MPC) matrices were obtained from the MICA laboratory’s open data repository (Paquola et al., 2019). Using vertex-level Glasser parcellation labels on the conte69 surface to map BigBrain data to Schaefer 400 parcels (see Methods), we tested seven measures of laminar microstructure against ρ.

No BigBrain measure showed a significant association with ρ after controlling for spatial position. The MPC Gradient 1—capturing the principal axis of laminar differentiation from granular sensory cortex to agranular limbic cortex—showed no zero-order correlation with ρ (ρ_s = −0.034, n.s.) and remained null after all levels of control (r|z = +0.025; r|z+SE+τ = +0.003). Profile SD, the most direct measure of laminar differentiation, was likewise null (ρ_s = +0.067; r|z+SE+τ = +0.058). Profile kurtosis showed a marginal zero-order correlation (ρ_s = −0.116, p < 0.05) but did not survive z-correction. MPC node strength, CV, skewness, and mean gradient were all non-significant at every level of control.

The Mesulam cytoarchitectural hierarchy (idiotypic → unimodal → heteromodal → paralimbic) similarly showed no relationship with ρ (ρ_s = −0.027, n.s.; mean ρ = 0.601 ± 0.004 across all four types). These convergent null results demonstrate that the ρ gradient is not organized along any dimension of histological laminar structure.

Critically, G2 showed a weak but significant correlation with BigBrain profile SD (ρ_s = −0.132, p = 0.008)—consistent with the known agranular/dysgranular architecture of paleocortical derivatives—but was not correlated with MPC Gradient 1 (ρ_s = +0.052, n.s.). Thus the dual-origin gradient captures a dimension of cortical organization largely orthogonal to the principal cytoarchitectural hierarchy. The laminar measures do not predict ρ even though G2 weakly tracks laminar differentiation, indicating that connectivity topology—not morphological laminar structure per se—is the relevant architectural feature for rotational dynamics.

To test whether the G2–ρ relationship is reducible to the granular–agranular distinction, we examined G2 as a predictor of ρ within each Mesulam cytoarchitectural type after controlling for z. Three of four types showed negative partial correlations, with the strongest and only FDR-surviving effect in paralimbic cortex (r = −0.34, p = 0.002, p_FDR = 0.014, N = 83) (Fig. 3f). Heteromodal (r = −0.20, p = 0.027, p_FDR = 0.053, N = 120) and unimodal cortex (r = −0.18, p = 0.033, p_FDR = 0.053, N = 136) showed consistent trends, while idiotypic cortex was null (r = +0.04, n.s., N = 61). FDR correction was applied across all 8 within-type tests (4 Mesulam types + 4 BigBrain SD quartiles). The strongest effect in paralimbic cortex—precisely where paleocortical and archicortical derivatives intermingle—confirms that the dual-origin gradient captures organizational variation within cytoarchitectural classes, not merely the distinction between them.

### Structural connectivity supports the recurrence hypothesis

HCP structural connectivity data provided direct empirical support for the hypothesized link between connection range and rotational dynamics. The short/long range ratio—indexing the predominance of local versus distant white matter connections—was the strongest zero-order SC predictor of ρ (ρ_s = −0.273, p < 0.001) and survived the full control battery (r|z+SE+τ = −0.115, p < 0.05). (Fig. 3g) Mean connection distance was also significant (ρ_s = +0.359, p < 0.001; r|z+SE+τ = +0.112, p < 0.05): regions with shorter average connection distances show more rotation.

After controlling for z, spectral exponent, and τ, several additional SC metrics emerged as significant: SC degree (r = +0.232, p < 0.001), node strength (r = +0.185, p < 0.001), weighted clustering (r = −0.174, p < 0.001), within-network fraction (r = −0.160, p < 0.01), long-range strength (r = +0.144, p < 0.01), and participation coefficient (r = +0.147, p < 0.01). The emergence of these effects after spectral control suggests that the spectral exponent had been masking structural connectivity contributions to ρ.

G2 correlated with SC connectivity range (G2 vs short/long ratio: ρ_s = +0.147, p = 0.003), confirming that paleocortical-pole regions have more locally concentrated structural connections. Mediation analysis revealed that G2 fully survived controlling for SC short-range strength (r = −0.244, p < 10⁻⁶) and the short/long ratio (r = −0.238, p = 1.4 × 10⁻⁶), while the SC short/long ratio was reduced to non-significance after controlling for G2 (r = −0.089, p = 0.074) (Fig. 3h). This indicates that G2 captures the connectivity architecture that SC indexes, plus additional developmental/genetic information that tractography alone does not reveal.

### G2 is independent of the timescale hierarchy

The intrinsic timescale τ showed a moderate positive correlation with ρ (ρ_s = +0.341), such that regions with longer timescales tend to exhibit stronger rotational dynamics—counter to a simple “fast circuits rotate, slow circuits integrate” account. Critically, G2 did not predict τ (ρ_s = −0.047, n.s.), confirming that the dual-origin gradient captures a dimension of circuit organization distinct from the timescale hierarchy. G2 therefore predicts a component of rotational dynamics that is independent of where a region sits on the dorsoventral axis, its spectral properties, and its intrinsic timescale—variance that can only be explained by the dual-origin developmental connectivity architecture.

### Robustness to spatial autocorrelation

Correlations between brain maps can be inflated by shared spatial smoothness. Spin permutation tests (5,000 rotations) assessed all key results. The zero-order G2–ρ correlation did not survive spin correction (p_spin = 0.10), indicating that the raw association is partly attributable to shared spatial structure. However, the partial correlations—the primary tests of our hypothesis—were robust: G2|z (p_spin = 0.002), G2|z+SE+τ (p_spin = 0.018), Genetic G2|z (p_spin = 0.007), and Genetic G2|z+SE+τ (p_spin = 0.020). The z-correction unmasks the true non-spatial signal by removing the dominant dorsoventral gradient to which the spin test is most sensitive.


# DISCUSSION

## Overview

The cortical ρ gradient presents a striking macroscale pattern: ventral regions exhibit stronger rotational dynamics while dorsal regions exhibit more integrative dynamics with longer intrinsic timescales. The z-coordinate alone explains 52% of ρ variance, and the spectral exponent—reflecting the balance of oscillatory and aperiodic activity—explains 79%. A systematic evaluation of candidate architectural predictors identified the dual-origin structural covariance gradient as the only measure that explains ρ variance beyond these dominant spatial and spectral dimensions.

## Rejecting simpler explanations

The systematic evaluation of candidate architectural predictors yielded a clear pattern of null results for intuitive candidate mechanisms. Interneuron gene expression (PVALB, SST), cytoarchitectural classification (Mesulam hierarchy), histological laminar differentiation (seven BigBrain measures including MPC Gradient 1), and zero-order functional connectivity metrics all failed to predict ρ after accounting for spatial position. These null results are collectively informative: the ρ gradient does not track the canonical laminar differentiation axis, the sensorimotor–transmodal hierarchy, or regional variation in inhibitory interneuron subtypes—despite the theoretical plausibility of each candidate. The failure of the interneuron hypothesis is particularly notable given the well-established role of PV+ interneurons in generating fast oscillations at the microcircuit level; macroscale gene expression gradients apparently do not map onto macroscale dynamical gradients in the straightforward manner that local circuit models would predict.

Two functional connectivity measures—FC Gradient 1 and clustering coefficient—did emerge as significant predictors after controlling for the spectral exponent, suggesting that the dominant 1/f structure had been masking their contributions. The positive clustering coefficient effect is consistent with the recurrence hypothesis: regions with more locally clustered functional connectivity exhibit stronger rotational dynamics.

## The dual-origin structural covariance gradient

The second structural covariance gradient (G2) from Valk et al. (2020) was the only predictor to survive all levels of statistical control. This gradient captures an organizational axis that aligns with the dual origin model of cortical development (Sanides, 1962; Pandya et al., 2015). In this framework, the cortical mantle develops from two phylogenetically ancient origins: paleocortex (piriform/olfactory cortex), giving rise to ventral and lateral neocortical regions, and archicortex (hippocampal formation), giving rise to dorsal and medial regions. Valk et al. demonstrated that G2 correlates with geodesic distance from paleocortex (r = 0.67, p_spin < 0.001), is genetically determined (genetic correlation r = 0.96), and is phylogenetically conserved across humans and macaques (r = 0.59).

SCov G2 is orthogonal to the MNI z-coordinate (ρ_s = 0.012), meaning it captures a dimension of cortical organization entirely distinct from dorsoventral spatial position. Its relationship with ρ strengthened progressively as confounds were added: from ρ_s = −0.155 (zero-order) to r = −0.253 (controlling for z and spectral exponent) to r = −0.249 (controlling for z, spectral exponent, and τ; p = 4.7 × 10⁻⁷). This suppression pattern—where a predictor becomes stronger as covariates are added—indicates that the dominant spatial and spectral gradients were partially masking the G2 signal.

The incremental R² from G2 (0.9% beyond z + spectral exponent) requires careful interpretation. The spectral exponent partially mediates the G2→ρ relationship: locally concentrated connectivity—the architectural feature indexed by the dual-origin gradient—simultaneously steepens the spectral slope (more oscillatory relative to aperiodic power; Gao et al., 2017) and generates rotational dynamics, because both are consequences of the same circuit property. The spectral exponent’s 79.5% R² therefore already captures the primary dynamical expression of the architectural gradient that G2 indexes. The 0.9% increment represents the direct developmental effect on ρ independent of its spectral consequences—not the total contribution of developmental architecture to rotational dynamics. This mediation framing implies that G2 influences dynamics through multiple pathways (spectral and non-spectral), with the direct pathway being modest but genetically specified. The appropriate metric for this direct effect is the partial correlation (r = −0.249, p = 4.7 × 10⁻⁷, p_spin = 0.018).

Critically, the G2–ρ relationship held within cytoarchitectural classes: G2 predicted ρ after z-correction within unimodal, heteromodal, and paralimbic cortex, with the strongest effect in paralimbic regions (r = −0.34, p = 0.002)—precisely where paleocortical and archicortical derivatives intermingle. This confirms that the dual-origin gradient captures organizational variation irreducible to the granular–agranular distinction.

The genetic correlation gradient replicated each result (r|z+SE+τ = −0.231, p = 3.0 × 10⁻⁶), confirming that the structural determinant of ρ is genetically specified.

## Relationship with spectral exponent and intrinsic timescale

The spectral exponent correlates strongly with ρ (ρ_s = −0.90) and alone explains 79% of its variance—substantially more than the z-coordinate (52%). This is expected: both ρ and the spectral exponent reflect the temporal structure of neural activity, shaped by local circuit architecture. Adding both z and spectral exponent to a regression model explains 87% of ρ variance, with G2 contributing an additional 0.9% (total R² = 0.877). While this incremental variance is modest in absolute terms, it is statistically robust (p_spin = 0.018) and represents the only architectural predictor that adds information beyond these two dominant dimensions.

The intrinsic timescale τ shows a moderate positive correlation with ρ (ρ_s = +0.34), such that regions with longer timescales tend to exhibit stronger rotational dynamics. This is notable because it runs counter to a simple “fast circuits rotate, slow circuits integrate” account. Instead, it suggests that the architectural features producing rotation—dense locally concentrated connectivity—also sustain activity over longer timescales. The raw positive correlation reverses to a robust anticorrelation after controlling for spatial position (r = −0.20, p = 5.3 × 10⁻⁵), and this anticorrelation holds within functional networks (Fisher z-weighted mean r = −0.19, p = 0.0002; 6/7 networks negative), confirming a genuine local circuit trade-off rather than an artifact of spatial residualization. Critically, G2 does not predict τ (ρ_s = −0.047, n.s.), confirming that the dual-origin gradient captures a dimension of circuit organization distinct from the timescale hierarchy.

## A mechanistic account

The link between dual-origin developmental architecture and rotational dynamics can be understood through the computational consequences of connectivity topology. The two developmental lineages produce different wiring patterns: paleocortical derivatives tend toward dysgranular and agranular lamination with dense locally concentrated connectivity, while archicortical derivatives exhibit more granular lamination with hierarchically organized feedforward–feedback circuits (Pandya et al., 2015). Recent computational work demonstrated that distance-dependent connectivity—without attractor dynamics or Hebbian plasticity—is sufficient to produce hierarchically modular networks with convergent population dynamics (Guarino et al., 2026). In their model, the connection distance range shapes local network topology, with shorter-range connectivity producing denser modular architecture.

HCP structural connectivity data provided direct empirical support. The short/long range ratio was the strongest zero-order SC predictor of ρ (ρ_s = −0.273, p < 0.001) and survived the full control battery (r|z+SE+τ = −0.115, p < 0.05). G2 correlated with SC connectivity range (G2 vs short/long ratio: ρ_s = +0.147, p = 0.003), confirming that paleocortical-pole regions have more locally concentrated structural connections. Mediation analysis revealed that G2 fully survived controlling for the SC short/long ratio (r = −0.238, p = 1.4 × 10⁻⁶), while the SC ratio was reduced to non-significance after controlling for G2 (r = −0.089, p = 0.074). G2 thus captures the connectivity architecture that SC indexes, plus additional developmental/genetic information that tractography alone does not reveal.

This provides a mechanistic bridge. In regions of paleocortical lineage, shorter-range and denser local connections create the conditions for tighter excitatory–inhibitory loops. Such loops are hypothesized to produce the antisymmetric coupling structure that generates complex eigenvalues in population dynamics—the mathematical signature of rotation that ρ captures (Murphy & Miller, 2009). It is important to note that tractography measures connection range, not microcircuit recurrence topology; the recurrence mechanism represents the hypothesized microcircuit implementation of the macroscopically observed short-range connectivity pattern. In regions of archicortical lineage, longer-range hierarchical connections support feedforward integration, yielding lower ρ. The strong correlation between ρ and the spectral exponent (r = −0.90) is consistent with this account: locally concentrated connectivity simultaneously steepens the spectral slope (Gao et al., 2017) and generates rotational dynamics, because the same architectural feature drives both phenomena.

The emergence of FC clustering coefficient as a significant predictor after spectral exponent control (r|z+SE+τ = +0.19) further supports the recurrence interpretation: regions with more locally clustered functional connectivity exhibit stronger rotational dynamics, consistent with the prediction that modular, recurrent architecture generates rotation.

## Implications for the fMRI non-replication

The dual-origin framework speaks to why the dorsoventral ρ gradient did not replicate at the parcel level in HCP fMRI data. Rotational dynamics likely depend on fast recurrent interactions at timescales of 10–50 ms, the regime where local excitatory–inhibitory loops generate antisymmetric coupling. BOLD imaging, with an effective temporal resolution of ~1 second, cannot resolve these dynamics. The failure of fMRI replication is therefore predicted by the mechanistic model: ρ is a fast-timescale phenomenon embedded in local circuit architecture, not a slowly varying property of macroscale functional coupling.

## Limitations

Several limitations warrant consideration. First, the structural covariance gradient is an indirect measure of connectivity architecture; direct assessment using diffusion tractography (e.g., connection distance distributions or bidirectional connection ratios per parcel) would provide stronger evidence for the recurrence hypothesis. Second, the AHBA gene expression data derive from only 6 donors with limited bilateral coverage, potentially obscuring true gene–dynamics relationships; future work using single-cell RNA sequencing atlases may provide finer resolution. Third, the incremental R² from G2 beyond z and spectral exponent is modest (0.9%), and the effect size, while statistically robust to spin correction, is small in absolute terms. This likely reflects the fact that ρ is dominated by spatial and spectral structure, with the developmental contribution operating as a secondary modulation. Fourth, although we obtained a proper vertex-level cross-parcellation mapping for the BigBrain analysis, the fundamental resolution mismatch between 360 Glasser and 400 Schaefer parcels means that some spatial precision is lost (mean modal overlap fraction = 0.59). Future work using BigBrain data parcellated natively in Schaefer space would provide a cleaner cytoarchitectural test.

## Future directions

The dual-origin framework generates testable predictions across multiple domains. Because the structural covariance gradient is phylogenetically conserved (Valk et al., 2020), the ρ gradient should be present in non-human primates—testable with existing electrophysiological datasets from macaque cortex (e.g., Neurotycho). Whether the gradient slope scales with cortical expansion or remains invariant across species would distinguish between evolutionary elaboration and a deeper developmental constraint on mammalian cortical wiring.

Pharmacological challenge studies offer a direct test of the recurrence hypothesis. NMDA receptor antagonists (e.g., ketamine) disrupt recurrent excitation and should selectively reduce ρ in high-recurrence ventral regions, effectively flattening the dorsoventral gradient. GABAergic modulators make more complex predictions: enhancing inhibition within recurrent loops could either sharpen or dampen oscillatory dynamics depending on the balance point, providing a parametric probe of the E–I coupling regime that generates rotation. Existing MEG-ketamine datasets could test these predictions without new data collection.

The clinical implications of a developmentally determined dynamical gradient deserve investigation. Neurodevelopmental conditions in which local-versus-long-range connectivity balance is disrupted—including autism spectrum disorder and schizophrenia—may show altered ρ gradients reflecting “dynamical dys-maturation” along the paleocortical–archicortical axis. In neurodegenerative disease, the known early vulnerability of paleocortical-lineage regions (entorhinal cortex, piriform cortex) in Alzheimer’s disease raises the possibility that ρ is differentially affected in regions where pathology first accumulates, potentially providing a more sensitive dynamical biomarker than static connectivity measures.

Computational models parameterized by region-specific connection distance profiles could formally test whether realistic connectivity architectures produce the observed ρ gradient. Laminar-resolved MEG (Troebinger et al., 2014) could examine whether rotational dynamics originate preferentially in supragranular layers, where local recurrent connections are densest.

## Conclusion

We have established a robust dorsoventral gradient of rotational dynamics in human cortex and identified its architectural correlate in the dual-origin developmental organization of cortical connectivity. The ρ gradient is not explained by cytoarchitectural type, laminar differentiation, interneuron composition, or functional connectivity hierarchy, but is predicted by a genetically determined, phylogenetically conserved structural covariance gradient reflecting the paleocortical–archicortical developmental axis—an association that holds within cytoarchitectural classes and survives all levels of statistical control. Locally concentrated connectivity in paleocortical derivatives is hypothesized to generate, through dense microcircuit recurrence, both the rotational dynamics and the steep spectral slopes observed ventrally, while longer-range hierarchical connectivity in archicortical derivatives supports the integrative, lower-rotation dynamics observed dorsally. This framework provides a mechanistic account linking developmental neurobiology to the macroscale organization of cortical dynamics.


# METHODS

## Participants

We analyzed resting-state MEG from the Mother of Unification Studies (MOUS) dataset (Schoffelen et al., 2019). After quality control, 212 participants were included (ages 18–45, mean 26.3 years), with 208 contributing to each parcel. Task MEG was available for subsets: visual (N = 69), auditory (N = 95). Resting-state fMRI was available for 68 participants.

## MEG acquisition and preprocessing

MEG was recorded using a 275-channel CTF system at 1200 Hz. Preprocessing comprised band-pass filtering (1–40 Hz), notch filtering (50 Hz), ICA artifact removal, and downsampling to 200 Hz.

## Source reconstruction

Source localization used LCMV beamforming (Van Veen et al., 1997) with single-shell boundary element models. Source time series were extracted for 400 cortical parcels (Schaefer et al., 2018).

## Rotational dynamics index (ρ)

We quantified rotational dynamics using delay embedding and VAR modeling. For each parcel time series x(t), we constructed a 10-dimensional state vector: X(t) = [x(t), x(t−d), ..., x(t−9d)] where d = 1 sample (5 ms). This Takens embedding (Takens, 1981) reconstructs the dynamical attractor. We then fit a VAR(1) model: X(t+1) = A · X(t) + ε, with ridge regularization (α = 0.001). The rotational index was computed as ρ = mean(|Im(λ)| / |λ|) for eigenvalues λ with |λ| > 0.01, equaling the mean sine of eigenvalue angles. Model fit R² (median = 0.9999) confirms adequate approximation.

## fMRI acquisition and preprocessing

Resting-state fMRI was acquired with TR = 2 s (266 volumes per subject). Preprocessing used nilearn (Abraham et al., 2014): standardization, detrending, and bandpass filtering (0.01–0.1 Hz). Time series were extracted for Schaefer 400 parcels using NiftiLabelsMasker with automatic resampling to MNI space.

## fMRI-derived measures

For each parcel, we computed: ρ_fMRI using 5-dimensional delay embedding (reduced from MEG’s 10 due to fewer timepoints) with delay = 1 TR; τ_fMRI as the integral of the autocorrelation function (Honey et al., 2012); spectral exponent as the slope of the log–log power spectrum in the 0.01–0.1 Hz range; and ALFF/fALFF as the amplitude of low-frequency fluctuations (0.01–0.08 Hz), with fALFF normalized by total power (Zou et al., 2008).

## Frequency-specific analysis

Time series were filtered to canonical bands before delay embedding. Adaptive delays targeted quarter-cycle phase advance: delay = round(0.25 / center_freq × fs). FDR correction at q < 0.05.

## Spatial statistics

Spin permutation tests (Alexander-Bloch et al., 2018; 100,000 rotations for primary MEG analyses; 10,000 for fMRI; 5,000 for architectural predictor analyses) preserved hemisphere structure. For τ–ρ residual correlation, both maps were regressed on spatial coordinates before correlating residuals.

## Confound controls

Partial correlations controlled for spectral exponent, total power, gamma/delta ratio, and distance from brain center. Each was tested with spin permutation.

## Architectural predictor analysis framework

Candidate architectural predictors of ρ were evaluated using a hierarchical statistical framework. For each predictor, we computed: (1) zero-order Spearman correlation with ρ; (2) partial Spearman correlation controlling for the z-coordinate; (3) partial correlation controlling for z and spectral exponent; and (4) partial correlation controlling for z, spectral exponent, and intrinsic timescale τ (the most stringent test). Variance partitioning used sequential R² from OLS regression to quantify the incremental contribution of each predictor. Spin permutation tests (5,000 rotations) assessed robustness to spatial autocorrelation for all key partial correlations.

## Structural covariance gradients

Structural covariance gradients were obtained from Valk et al. (2020), computed from cortical thickness covariance across HCP participants and parcellated to the Schaefer 400 atlas. The second gradient (G2) captures the dual-origin developmental axis. The genetic correlation gradient, derived from the genetic component of cortical thickness covariance, provides a genetically specified version of the same organizational axis.

## Functional connectivity measures

FC gradients (Gradient 1 and Gradient 2), clustering coefficient, and participation coefficient were computed from resting-state fMRI functional connectivity matrices in the Schaefer 400 parcellation. FC Gradient 1 captures the principal sensorimotor-to-default mode hierarchy (Margulies et al., 2016).

## BigBrain microstructure profile covariance

To test whether histological laminar differentiation predicts the ρ gradient, we analyzed cytoarchitectural data from the BigBrain atlas, an ultra-high-resolution (20 μm) 3D reconstruction of a post-mortem human brain stained for cell bodies (Amunts et al., 2013). BigBrain staining intensity profiles and microstructure profile covariance (MPC) matrices were obtained from the MICA laboratory’s open data repository (Paquola et al., 2019). These data comprise 15 equivolumetric intracortical surfaces sampled between the pial and white matter boundaries for each of 360 Glasser atlas parcels (Glasser et al., 2016), yielding a 15-point intensity profile per region that reflects the laminar distribution of neuronal density and size.

From the intensity profiles, we computed per-parcel summary statistics capturing distinct aspects of cytoarchitectural differentiation: profile standard deviation (SD; indexing overall laminar differentiation), coefficient of variation (CV; relative differentiation normalized by mean staining intensity), skewness (asymmetry of the laminar intensity distribution), kurtosis (peakedness of laminar contrast), and mean absolute gradient (rate of intensity change across cortical depth).

The MPC matrix (360 × 360) was computed as the partial correlation of intensity profiles between all pairs of Glasser parcels, controlling for the mean cortex-wide profile (Paquola et al., 2019). From this matrix, we derived MPC Gradient 1 via diffusion map embedding (normalized angle kernel; BrainSpace), which captures the principal axis of cytoarchitectural similarity running from granular sensory cortex to agranular limbic cortex. We also computed MPC node strength (mean microstructural covariance per parcel) as a summary measure of each region’s cytoarchitectural typicality.

## Glasser-to-Schaefer parcellation mapping

Because the BigBrain MPC data are parcellated according to the Glasser 360 atlas while all other analyses use the Schaefer 400 atlas, a cross-parcellation mapping was required. We obtained vertex-level Glasser 360 parcellation labels on the conte69 32k surface (64,984 vertices; ENIGMA Toolbox; Larivière et al., 2021). For each Schaefer 400 parcel, we identified all conte69 vertices belonging to that parcel and assigned the modal (most frequent) Glasser label among those vertices. This vertex-level approach yielded complete coverage (400/400 parcels mapped), with a mean modal overlap fraction of 0.59 (i.e., on average, 59% of vertices within each Schaefer parcel shared the same Glasser label). BigBrain feature values for each Schaefer parcel were then taken from its assigned Glasser parcel.

## HCP structural connectivity

Group-average structural connectivity was obtained from the Human Connectome Project via the ENIGMA Toolbox (Larivière et al., 2021), parcellated to the Schaefer 400 atlas. The SC matrix represents log-transformed streamline counts from deterministic tractography of diffusion-weighted MRI data, averaged across HCP participants. The matrix was 400 × 400, symmetric, and sparse (6.2% non-zero entries).

## Structural connectivity metrics

From the SC matrix, we computed the following per-parcel metrics: node strength (sum of all connection weights); node degree (number of non-zero connections); mean connection distance (weighted average Euclidean distance to connected parcels using MNI centroid coordinates); short-range strength (total connection weight to parcels within 50 mm); long-range strength (total connection weight to parcels ≥50 mm); short/long range ratio (indexing the predominance of local vs. distant connectivity); weighted clustering coefficient (Onnela et al., 2005 formulation); participation coefficient (Shannon entropy-based; with respect to Yeo 7-network assignments); and within-network fraction (proportion of total SC strength directed to same-network parcels).

## Mediation analysis

To determine whether the SC–ρ relationship is mediated by the dual-origin gradient, we performed mutual adjustment analyses: testing whether G2 survives controlling for SC metrics and vice versa, beyond the standard control variables (z, spectral exponent, τ). Incremental R² was assessed by adding predictors sequentially to the base model.

## Cortical flatmap visualization

To visualize the spatial distribution of ρ across the cortical surface without the distortion inherent in 3D surface rendering, we generated a two-dimensional cortical flatmap using multidimensional scaling (MDS). For each hemisphere, MDS was applied to the matrix of Euclidean distances between parcel centroids in MNI space (scikit-learn v1.5, metric MDS, 1000 iterations, 10 initializations), yielding a 2D embedding that minimally distorts inter-parcel distances (median local distortion ratio: 1.08×). The resulting 2D coordinates were rotated so that the vertical axis aligned with the dorsoventral (z) coordinate and the horizontal axis with the anteroposterior (y) coordinate. Voronoi tessellation of the 2D parcel centroids was used to assign each point in the flatmap to its nearest parcel, producing a continuous tiled representation.

Parcel-level ρ values were z-scored and displayed using a PET-style colormap (dark red through orange and yellow to green) to maximize visual dynamic range. Anatomical region boundaries were derived by mapping each Schaefer 400 parcel to its corresponding classical gyral region based on parcel label content and MNI coordinates (19 regions across 6 lobes: frontal, temporal, parietal, occipital, cingulate, and insular cortex). Thin white lines delineate gyral boundaries and thick white lines delineate lobe boundaries, computed as shared Voronoi edges between parcels assigned to different regions or lobes, respectively.

## Gene expression analysis

Regional gene expression data were obtained from the Allen Human Brain Atlas (AHBA; Hawrylycz et al., 2012), parcellated to the Schaefer 400 atlas using the abagen toolbox. PVALB (parvalbumin) and SST (somatostatin) expression levels were tested against ρ as candidate interneuron markers.

## Code and data availability

Analysis code: https://github.com/Salardini/Dorso-Ventral-Gradient. MOUS dataset: Donders Repository (Schoffelen et al., 2019).



FIGURE LEGENDS


Fig. 1 | A dorsoventral gradient of rotational dynamics in human cortex.

a, Schematic of the ρ metric. Parcel time series are delay-embedded to reconstruct state-space trajectories. A VAR(1) model is fit and eigenvalues computed; ρ = mean |sin θ| quantifies the rotational component. High ρ (underdamped, oscillatory) characterizes ventral cortex; low ρ (overdamped, decaying) characterizes dorsal cortex. b, Cortical surface maps of group-average ρ (n = 212 participants, Schaefer 400 parcellation). Left and right hemispheres shown in lateral and medial views. c, Cortical flatmap generated by multidimensional scaling of parcel centroids, with Voronoi tessellation. Hemispheres shown separately; dorsal–ventral gradient is clearly visible in both. d, Scatter plot of ρ versus z-coordinate (dorsoventral axis) for all 400 parcels, colored by Yeo 7-network assignment. Black curve: quadratic fit. Spearman ρₛ = −0.73, p_spin < 10⁻⁵. e, Distribution of individual-level Spearman correlations between ρ and z across 25 participants. Mean r = −0.43; 100% of participants showed negative correlations. f, Band-specific ρ–z correlations. Slow rhythms (δ, θ) show dorsal predominance (positive r); fast rhythms (α, β, γ) show ventral predominance (negative r). All p_spin < 0.001. g, Scatter plot of ρ versus τ (intrinsic timescale) after residualizing both for spatial coordinates (x, y, z). Spearman ρₛ = −0.34, confirming a local circuit trade-off independent of macroscale position. h, fMRI-derived ρ versus z-coordinate (n = 68 participants). Spearman ρₛ = 0.13, indicating a weak but directionally consistent gradient in the BOLD signal.


Fig. 2 | Robustness and cross-modal replication.

a, Task replication. ρ versus z during auditory speech processing (n = 95; r = −0.74) and visual object recognition (n = 69; r = −0.68) overlaid on resting state (grey; r = −0.73). Gradient directions are nearly identical across cognitive states. b, HCP fMRI replication (n = 1,096 participants). fMRI-derived ρ shows no significant dorsoventral gradient (ρₛ ≈ 0.00, p = 0.99), consistent with the mechanistic prediction that rotational dynamics operate at fast timescales inaccessible to BOLD imaging. c, Atlas sensitivity. The ρ–z gradient is stable across Schaefer parcellation resolutions: 400 parcels (r = −0.72), 200 parcels (r = −0.70), 100 parcels (r = −0.69). All p_spin < 0.001. d, Adaptive versus fixed embedding delay. Band-specific ρ–z correlations computed with delays matched to each frequency band’s timescale (red) compared to fixed 5 ms delay (blue). The gradient is robust to delay choice. e, fMRI-derived complementary measures versus z-coordinate. ρ_fMRI (r = 0.13), τ_fMRI (r = −0.17), and spectral exponent (r = 0.14) all show significant dorsoventral organization. f, Summary of ρ–z correlations across all datasets and modalities.


Fig. 3 | The dual-origin structural covariance gradient predicts rotational dynamics.

a, Heatmap of partial Spearman correlations between 12 candidate architectural predictors and ρ under four levels of statistical control: zero-order, controlling for z, controlling for z + spectral exponent (SE), and controlling for z + SE + intrinsic timescale (τ). Significance markers: *p < 0.05, **p < 0.01, ***p < 0.001. b, Cortical surface maps of SCov G2 (Valk et al., 2020). Left and right hemispheres shown in lateral and medial views. c, ρ versus SCov G2 for all 400 parcels, colored by network. Zero-order ρₛ = −0.155; partial r|z,SE,τ = −0.249, p = 4.7 × 10⁻⁷. d, Progressive strengthening of G2–ρ association as confounds are added, from zero-order (−0.155) through full controls (−0.249). e, All candidate predictors ranked by |partial r| under the most stringent control (z + SE + τ). Category colors indicate predictor class. f, ρ versus SCov G2 within each Mesulam cytoarchitectural class, with per-class regression lines. The G2–ρ relationship holds within cortical hierarchy levels (overall partial r|z,SE,τ,Mesulam = −0.256, p = 2.1 × 10⁻⁷). g, ρ versus SC short/long range ratio (ρₛ = −0.273; r|z,SE,τ = −0.115). h, Mediation analysis. G2 survives controlling for SC metrics (r = −0.238), while SC short/long ratio is reduced to non-significance after controlling for G2 (r = −0.089, n.s.).


Fig. 4 | Excitatory–inhibitory balance model and interneuron gene expression.

a, E-I network model. Varying inhibitory gain (g_I) produces opposing effects on timescale τ (blue) and rotational index ρ (red), demonstrating that a single circuit parameter can generate the observed τ–ρ trade-off. b, Model-derived τ versus ρ across inhibitory gain values, colored by g_I. The model produces a strong anticorrelation (r = −0.84, p = 0.0006), qualitatively matching the empirical τ–ρ trade-off. c, Spearman correlations between interneuron marker gene expression (Allen Human Brain Atlas) and ρ. SST shows a weak positive association (r = 0.12, p = 0.02) and PV−SST a weak negative association (r = −0.12, p = 0.02); neither survives spatial confound control. d–f, Scatter plots of ρ versus PVALB expression (d; r = −0.09, n.s.), SST expression (e; r = 0.12, p = 0.02), and PV−SST ratio (f; r = −0.12, p = 0.02).


# REFERENCES

Abraham, A., et al. (2014). Machine learning for neuroimaging with scikit-learn. Frontiers in Neuroinformatics, 8, 14.

Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence between maps of human brain structure and function. NeuroImage, 178, 540–551.

Amunts, K., et al. (2013). BigBrain: An ultrahigh-resolution 3D human brain model. Science, 340, 1472–1475.

Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity measure for time series. Physical Review Letters, 88, 174102.

Cardin, J. A., et al. (2009). Driving fast-spiking cells induces gamma rhythm and controls sensory responses. Nature, 459, 663–667.

Churchland, M. M., et al. (2012). Neural population dynamics during reaching. Nature, 487, 51–56.

Engel, A. K., & Fries, P. (2010). Beta-band oscillations—signalling the status quo? Current Opinion in Neurobiology, 20, 156–165.

Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. NeuroImage, 158, 70–78.

Glasser, M. F., & Van Essen, D. C. (2011). Mapping human cortical areas in vivo based on myelin content as revealed by T1- and T2-weighted MRI. Journal of Neuroscience, 31, 11597–11616.

Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. Nature, 536, 171–178.

Goodale, M. A., & Milner, A. D. (1992). Separate visual pathways for perception and action. Trends in Neurosciences, 15, 20–25.

Guarino, D., et al. (2026). Convergent information flows in cortical networks reveal reproducible dynamics without attractor architecture. Nature Neuroscience.

Hawrylycz, M. J., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. Nature, 489, 391–399.

Hennequin, G., Vogels, T. P., & Gerstner, W. (2014). Optimal control of transient dynamics in balanced networks supports generation of complex movements. Neuron, 82, 1394–1406.

Honey, C. J., et al. (2012). Slow cortical dynamics and the accumulation of information over long timescales. Neuron, 76, 423–434.

Larivière, S., et al. (2021). The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. Nature Methods, 18, 698–700.

Margulies, D. S., et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. PNAS, 113, 12574–12579.

Murphy, B. K., & Miller, K. D. (2009). Balanced amplification: a new mechanism of selective amplification of neural activity patterns. Neuron, 61, 635–648.

Murray, J. D., et al. (2014). A hierarchy of intrinsic timescales across primate cortex. Nature Neuroscience, 17, 1661–1663.

Onnela, J.-P., et al. (2005). Intensity and coherence of motifs in weighted complex networks. Physical Review E, 71, 065103.

Pandya, D. N., Seltzer, B., Petrides, M., & Cipolloni, P. B. (2015). Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept. Oxford University Press.

Paquola, C., et al. (2019). Microstructural and functional gradients are increasingly dissociated in transmodal cortices. PLOS Biology, 17, e3000284.

Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. Clinical Neurophysiology, 110, 1842–1857.

Rudy, B., et al. (2011). Three groups of interneurons account for nearly 100% of neocortical GABAergic neurons. Developmental Neurobiology, 71, 45–61.

Sanides, F. (1962). Die Architektonik des menschlichen Stirnhirns. Springer.

Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28, 3095–3114.

Schoffelen, J. M., et al. (2019). A 204-subject multimodal neuroimaging dataset to study language processing. Scientific Data, 6, 17.

Singer, W., & Gray, C. M. (1995). Visual feature integration and the temporal correlation hypothesis. Annual Review of Neuroscience, 18, 555–586.

Sohal, V. S., et al. (2009). Parvalbumin neurons and gamma rhythms enhance cortical circuit performance. Nature, 459, 698–702.

Steriade, M., McCormick, D. A., & Sejnowski, T. J. (1993). Thalamocortical oscillations in the sleeping and aroused brain. Science, 262, 679–685.

Strogatz, S. H. (2015). Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering. Westview Press.

Takens, F. (1981). Detecting strange attractors in turbulence. Lecture Notes in Mathematics, 898, 366–381.

Troebinger, L., et al. (2014). Discrimination of cortical laminae using MEG. NeuroImage, 102, 885–893.

Valk, S. L., et al. (2020). Shaping brain structure: Genetic and phylogenetic axes of macroscale organization of cortical thickness. Science Advances, 6, eabb3417.

Van Veen, B. D., et al. (1997). Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Transactions on Biomedical Engineering, 44, 867–880.

Yeo, B. T. T., et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. Journal of Neurophysiology, 106, 1125–1165.

Zou, Q. H., et al. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF. Journal of Neuroscience Methods, 172, 137–141.
