# Discussion: Architectural Origins of the Rotational Dynamics Gradient

## Overview

The cortical ρ gradient presents a striking macroscale pattern: ventral regions exhibit stronger rotational dynamics while dorsal regions exhibit more integrative dynamics with longer intrinsic timescales. The z-coordinate alone explains 52% of ρ variance, and the spectral exponent — reflecting the balance of oscillatory and aperiodic activity — explains 79%. A systematic evaluation of candidate architectural predictors identified the dual-origin structural covariance gradient as the only measure that explains ρ variance beyond these dominant spatial and spectral dimensions.

## Rejecting simpler explanations

Several intuitive candidate mechanisms failed to predict ρ after accounting for spatial position and spectral structure.

The interneuron hypothesis — that regional variation in fast-spiking parvalbumin-positive (PV+) interneurons drives the ρ gradient through differential sharpening of oscillatory dynamics — was not supported. PVALB expression showed no significant correlation with ρ (ρₛ = −0.09, p = 0.11), and SST expression showed a weak association in the wrong direction (ρₛ = +0.12, p = 0.02). Both were null after controlling for z (p > 0.8). While E-I balance is clearly relevant to rotational dynamics at the microcircuit level (Churchland et al., 2012; Murphy & Miller, 2009), macroscale variation in interneuron gene expression does not account for the cortical ρ gradient.

Functional connectivity metrics were similarly uninformative. FC Gradient 1 (the sensorimotor-to-default mode hierarchy; Margulies et al., 2016) showed no zero-order correlation with ρ and only a marginal partial effect after z-correction (r|z = −0.11, p = 0.03). FC Gradient 2, clustering coefficient, and participation coefficient were all non-significant after z-correction. However, after additionally controlling for the spectral exponent, FC Gradient 1 and clustering coefficient both emerged as significant predictors (r|z+SE = −0.24 and +0.20, respectively), suggesting that these metrics capture variance in ρ that is masked by the dominant 1/f structure but becomes visible once it is removed.

The Mesulam cytoarchitectural hierarchy (idiotypic → unimodal → heteromodal → paralimbic) showed no relationship with ρ in any model (ρₛ = −0.027, n.s.; mean ρ effectively identical across all four types at 0.601 ± 0.004). This null result is informative: the ρ gradient does not track the canonical laminar differentiation axis that distinguishes primary sensory cortex from limbic cortex.

BigBrain histological data provided a more fine-grained test of the cytoarchitectural hypothesis. Using vertex-level Glasser parcellation labels on the conte69 surface to map BigBrain staining intensity profiles to Schaefer 400 parcels, we tested seven measures of laminar microstructure against ρ: MPC Gradient 1 (the principal sensory-to-limbic cytoarchitectural axis), MPC node strength, profile SD (laminar differentiation), CV, skewness, kurtosis, and mean gradient. None predicted ρ after z-correction (all |r| < 0.06, n.s.). Profile kurtosis showed a marginal zero-order correlation (ρₛ = −0.116, p < 0.05) that did not survive spatial control. This comprehensive null result, obtained with a proper cross-parcellation mapping, demonstrates that the ρ gradient is not organized along any dimension of histological laminar structure as captured by BigBrain staining intensity.

## The dual-origin structural covariance gradient

The second structural covariance gradient (G2) from Valk et al. (2020) was the only predictor to survive all levels of statistical control. This gradient, computed from cortical thickness covariance across HCP participants in the same Schaefer 400 parcellation used here, captures an organizational axis that aligns with the dual origin model of cortical development (Sanides, 1962; Pandya et al., 2015). In this framework, the cortical mantle develops from two phylogenetically ancient origins: paleocortex (piriform/olfactory cortex), giving rise to ventral and lateral neocortical regions through successive waves of laminar elaboration, and archicortex (hippocampal formation), giving rise to dorsal and medial regions. Valk et al. demonstrated that G2 correlates with geodesic distance from paleocortex (r = 0.67, p_spin < 0.001), is genetically determined (genetic correlation r = 0.96), and is phylogenetically conserved across humans and macaques (r = 0.59).

SCov G2 is orthogonal to the MNI z-coordinate (ρₛ = 0.012), meaning it captures a dimension of cortical organization entirely distinct from dorsoventral spatial position. Its relationship with ρ strengthened progressively as confounds were added: from ρₛ = −0.155 (zero-order) to r = −0.213 (controlling for z) to r = −0.253 (controlling for z and spectral exponent) to r = −0.249 (controlling for z, spectral exponent, and τ; p = 4.7 × 10⁻⁷). This pattern — where a predictor becomes stronger rather than weaker as covariates are added — indicates that the dominant spatial and spectral gradients were partially masking the G2 signal. Removing them reveals the underlying developmental contribution more clearly.

The genetic correlation gradient replicated each result (r|z+SE+τ = −0.231, p = 3.0 × 10⁻⁶), confirming that the structural determinant of ρ is genetically specified. No other structural covariance gradient showed significant partial correlations with ρ.

## Robustness to spatial autocorrelation

Correlations between brain maps can be inflated by shared spatial smoothness. To address this, we performed spin permutation tests (Alexander-Bloch et al., 2018; 5,000 rotations) for all key results. The zero-order G2–ρ correlation did not survive spin correction (p_spin = 0.10), indicating that the raw association is partly attributable to shared spatial structure. However, the partial correlations — which are the primary tests of our hypothesis — were robust: G2|z (p_spin = 0.002), G2|z+SE+τ (p_spin = 0.018), Genetic G2|z (p_spin = 0.007), and Genetic G2|z+SE+τ (p_spin = 0.020). The z-correction does not merely remove a confound; it unmasks the true non-spatial signal by removing the dominant dorsoventral gradient to which the spin test is most sensitive.

## Relationship with spectral exponent and intrinsic timescale

The spectral exponent correlates strongly with ρ (ρₛ = −0.90) and alone explains 79% of its variance — substantially more than the z-coordinate (52%). This is expected: both ρ and the spectral exponent reflect the temporal structure of neural activity, which is shaped by local circuit architecture. Adding both z and spectral exponent to a regression model explains 87% of ρ variance, with G2 contributing an additional 0.9% (total R² = 0.877). While this incremental variance is modest in absolute terms, it is statistically robust (p_spin = 0.018) and represents the only architectural predictor that adds information beyond these two dominant dimensions.

The intrinsic timescale τ shows a moderate positive correlation with ρ (ρₛ = +0.34), such that regions with longer timescales tend to exhibit stronger rotational dynamics. This is notable because it runs counter to a simple "fast circuits rotate, slow circuits integrate" account. Instead, it suggests that the architectural features producing rotation — dense local recurrent connectivity — also sustain activity over longer timescales. Critically, G2 does not predict τ (ρₛ = −0.047, n.s.), confirming that the dual-origin gradient captures a dimension of circuit organization distinct from the timescale hierarchy.

## A mechanistic account

The link between dual-origin developmental architecture and rotational dynamics can be understood through the computational consequences of connectivity topology. The two developmental lineages produce different wiring patterns: paleocortical derivatives tend toward dysgranular and agranular lamination with dense local recurrent connectivity, while archicortical derivatives exhibit more granular lamination with hierarchically organized feedforward-feedback circuits (Pandya et al., 2015). Recent computational work demonstrated that distance-dependent connectivity — without attractor dynamics or Hebbian plasticity — is sufficient to produce hierarchically modular networks with convergent population dynamics (Guarino et al., 2026). In their model, the connection distance range shapes local network topology, with shorter-range connectivity producing denser modular architecture.

HCP structural connectivity data provided direct empirical support for this account. The short/long range ratio — indexing the predominance of local versus distant white matter connections — was the strongest zero-order SC predictor of ρ (ρₛ = −0.273, p < 0.001) and survived the full control battery (r|z+SE+τ = −0.115, p < 0.05). Crucially, G2 correlated with SC connectivity range (G2 vs short/long ratio: ρₛ = +0.147, p = 0.003), confirming that paleocortical-pole regions have more locally concentrated structural connections. Mediation analysis revealed that G2 fully survived controlling for SC short-range strength (r = −0.244, p < 10⁻⁶), while the SC short/long ratio was reduced to non-significance after controlling for G2 (r = −0.089, p = 0.074). This indicates that G2 captures the connectivity architecture that SC indexes, plus additional developmental/genetic information that tractography alone does not reveal.

This provides a direct mechanistic bridge. In regions of paleocortical lineage, shorter-range and denser local connections create tighter recurrent excitatory-inhibitory loops. These loops produce the antisymmetric coupling structure that generates complex eigenvalues in population dynamics — the mathematical signature of rotation that ρ captures (Murphy & Miller, 2009). In regions of archicortical lineage, longer-range hierarchical connections support feedforward integration, yielding lower ρ. The strong correlation between ρ and the spectral exponent (r = −0.90) is consistent rather than redundant: local recurrence simultaneously steepens the spectral slope (more oscillatory relative to aperiodic power; Gao et al., 2017) and generates rotational dynamics, because the same architectural feature — dense local recurrence — drives both phenomena.

The emergence of FC clustering coefficient as a significant predictor after spectral exponent control (r|z+SE+τ = +0.19, p_spin to be determined) further supports the recurrence interpretation: regions with more locally clustered functional connectivity exhibit stronger rotational dynamics, consistent with the prediction that modular, recurrent architecture generates rotation.

## Implications for the fMRI non-replication

The dual-origin framework speaks to why the dorsoventral ρ gradient did not replicate in HCP fMRI data (main text). Rotational dynamics likely depend on fast recurrent interactions at timescales of 10–50 ms, the regime where local E-I loops generate antisymmetric coupling. BOLD imaging, with an effective temporal resolution of ~1 second, cannot resolve these dynamics. The failure of fMRI replication is therefore predicted by the mechanistic model: ρ is a fast-timescale phenomenon embedded in local circuit architecture, not a slowly varying property of macroscale functional coupling.

## Limitations

Several limitations warrant consideration. First, the structural covariance gradient is an indirect measure of connectivity architecture; direct assessment using diffusion tractography (e.g., connection distance distributions or bidirectional connection ratios per parcel) would provide stronger evidence for the recurrence hypothesis. Second, the AHBA gene expression data derive from only 6 donors with limited bilateral coverage, potentially obscuring true gene–dynamics relationships; future work using single-cell RNA sequencing atlases may provide finer resolution. Third, the incremental R² from G2 beyond z and spectral exponent is modest (0.9%), and the effect size, while statistically robust to spin correction, is small in absolute terms. This likely reflects the fact that ρ is dominated by spatial and spectral structure, with the developmental contribution operating as a secondary modulation. Fourth, although we obtained a proper vertex-level cross-parcellation mapping for the BigBrain analysis (via Glasser conte69 labels from the ENIGMA Toolbox), the fundamental resolution mismatch between 360 Glasser and 400 Schaefer parcels means that some spatial precision is lost in the mapping (mean modal overlap fraction = 0.59). Future work using BigBrain data parcellated natively in Schaefer space would provide a cleaner cytoarchitectural test.

## Future directions

The dual-origin framework generates testable predictions. Because the structural covariance gradient is phylogenetically conserved (Valk et al., 2020), the ρ gradient should be present in non-human primates — testable with existing electrophysiological datasets from macaque cortex. Computational models parameterized by region-specific connection distance profiles could formally test whether realistic connectivity architectures produce the observed ρ gradient. Laminar-resolved MEG (Troebinger et al., 2014) could examine whether rotational dynamics originate preferentially in supragranular layers, where local recurrent connections are densest.

## References

Alexander-Bloch, A. F., et al. (2018). On testing for spatial correspondence between maps of human brain structure and function. *NeuroImage,* 178, 540–551.

Churchland, M. M., et al. (2012). Neural population dynamics during reaching. *Nature,* 487, 51–56.

Gao, R., Peterson, E. J., & Voytek, B. (2017). Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage,* 158, 70–78.

Guarino, D., et al. (2026). Convergent information flows in cortical networks reveal reproducible dynamics without attractor architecture. *Nature Neuroscience.*

Hawrylycz, M. J., et al. (2012). An anatomically comprehensive atlas of the adult human brain transcriptome. *Nature,* 489, 391–399.

Margulies, D. S., et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *PNAS,* 113, 12574–12579.

Murphy, B. K., & Miller, K. D. (2009). Balanced amplification: a new mechanism of selective amplification of neural activity patterns. *Neuron,* 61, 635–648.

Pandya, D. N., Seltzer, B., Petrides, M., & Cipolloni, P. B. (2015). *Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept.* Oxford University Press.

Sanides, F. (1962). *Die Architektonik des menschlichen Stirnhirns.* Springer.

Troebinger, L., et al. (2014). Discrimination of cortical laminae using MEG. *NeuroImage,* 102, 885–893.

Valk, S. L., et al. (2020). Shaping brain structure: Genetic and phylogenetic axes of macroscale organization of cortical thickness. *Science Advances,* 6, eabb3417.
