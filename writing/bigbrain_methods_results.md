# BigBrain Cytoarchitectural Analysis: Methods and Results

## Methods

### BigBrain microstructure profile covariance

To test whether histological laminar differentiation predicts the ρ gradient, we analyzed cytoarchitectural data from the BigBrain atlas, an ultra-high-resolution (20 μm) 3D reconstruction of a post-mortem human brain stained for cell bodies (Amunts et al., 2013). BigBrain staining intensity profiles and microstructure profile covariance (MPC) matrices were obtained from the MICA laboratory's open data repository (Paquola et al., 2019; https://github.com/MICA-MNI/micaopen). These data comprise 15 equivolumetric intracortical surfaces sampled between the pial and white matter boundaries for each of 360 Glasser atlas parcels (Glasser et al., 2016), yielding a 15-point intensity profile per region that reflects the laminar distribution of neuronal density and size.

From the intensity profiles, we computed per-parcel summary statistics capturing distinct aspects of cytoarchitectural differentiation: profile standard deviation (SD; indexing overall laminar differentiation, with higher values in granular/koniocortex and lower values in agranular/dysgranular cortex), coefficient of variation (CV; relative differentiation normalized by mean staining intensity), skewness (asymmetry of the laminar intensity distribution), kurtosis (peakedness of laminar contrast), and mean absolute gradient (rate of intensity change across cortical depth).

The MPC matrix (360 × 360) was computed as the partial correlation of intensity profiles between all pairs of Glasser parcels, controlling for the mean cortex-wide profile (Paquola et al., 2019). From this matrix, we derived MPC Gradient 1 via diffusion map embedding (normalized angle kernel; BrainSpace), which captures the principal axis of cytoarchitectural similarity running from granular sensory cortex to agranular limbic cortex. We also computed MPC node strength (mean microstructural covariance per parcel) as a summary measure of each region's cytoarchitectural typicality.

### Glasser-to-Schaefer parcellation mapping

Because the BigBrain MPC data are parcellated according to the Glasser 360 atlas while all other analyses use the Schaefer 400 atlas, a cross-parcellation mapping was required. We obtained vertex-level Glasser 360 parcellation labels on the conte69 32k surface (64,984 vertices; ENIGMA Toolbox; Larivière et al., 2021). For each Schaefer 400 parcel, we identified all conte69 vertices belonging to that parcel and assigned the modal (most frequent) Glasser label among those vertices. This vertex-level approach yielded complete coverage (400/400 parcels mapped), with a mean modal overlap fraction of 0.59 (i.e., on average, 59% of vertices within each Schaefer parcel shared the same Glasser label). BigBrain feature values for each Schaefer parcel were then taken from its assigned Glasser parcel.

This approach replaces a previous approximate mapping based on KMeans spatial clustering of conte69 vertex coordinates, which produced effectively random Glasser label assignments (validation: BigBrain profile SD vs z-coordinate ρₛ = −0.016, n.s.). The vertex-level mapping, while still constrained by the fundamental resolution mismatch between 360 Glasser and 400 Schaefer parcels (many-to-many relationships exist, particularly for small parcels), provides the most accurate cross-parcellation alignment achievable without native Schaefer-space BigBrain data.

### Statistical analysis of BigBrain measures

All BigBrain features were tested against ρ using the same statistical framework applied to other predictors: zero-order Spearman correlations, partial Spearman correlations controlling for z, and partial correlations controlling for z + spectral exponent + τ (the most stringent test). We also assessed the relationship between BigBrain measures and the structural covariance gradient G2 to determine whether the dual-origin gradient's predictive power might be mediated by laminar differentiation.

## Results

### BigBrain laminar features do not predict ρ

No BigBrain cytoarchitectural measure showed a significant association with ρ after controlling for spatial position (Table 1). The MPC Gradient 1 — which captures the principal axis of laminar differentiation from granular sensory cortex to agranular limbic cortex — showed no zero-order correlation with ρ (ρₛ = −0.034, n.s.) and remained null after all levels of control (r|z = +0.025, n.s.; r|z+SE+τ = +0.003, n.s.). Profile SD, the most direct measure of laminar differentiation, was likewise null (ρₛ = +0.067, n.s.; r|z+SE+τ = +0.058, n.s.). Profile kurtosis showed a marginal zero-order correlation (ρₛ = −0.116, p < 0.05) but did not survive z-correction (r|z = −0.051, n.s.). MPC node strength, profile CV, skewness, and mean gradient were all non-significant at every level of control.

**Table 1. BigBrain cytoarchitectural features vs ρ (n = 400 parcels)**

| Feature | ρₛ | r\|z | r\|z+SE+τ |
|---|---|---|---|
| MPC Gradient 1 | −0.034 | +0.025 | +0.003 |
| MPC strength | −0.016 | −0.054 | +0.052 |
| Profile SD | +0.067 | +0.058 | +0.058 |
| Profile CV | +0.057 | +0.042 | +0.057 |
| Profile skewness | +0.067 | −0.043 | −0.016 |
| Profile kurtosis | −0.116* | −0.051 | −0.033 |
| Profile gradient | +0.044 | +0.020 | +0.034 |

\* p < 0.05; all other correlations n.s.

### Dissociation between laminar differentiation and the dual-origin gradient

The structural covariance gradient G2 showed a weak but significant correlation with BigBrain profile SD (ρₛ = −0.132, p = 0.008), indicating that regions at the paleocortical end of the dual-origin axis tend to have lower laminar differentiation — consistent with the known agranular/dysgranular architecture of paleocortical derivatives. However, G2 was not correlated with MPC Gradient 1 (ρₛ = +0.052, n.s.), suggesting that the developmental gradient captures a dimension of cortical organization that is largely orthogonal to the principal cytoarchitectural hierarchy.

This dissociation is critical: although G2 predicts ρ robustly (r|z+SE+τ = −0.249, p_spin = 0.018) and weakly tracks laminar differentiation (via profile SD), the laminar measures themselves do not predict ρ. The dual-origin gradient therefore captures an aspect of connectivity architecture — likely the density and range of local recurrent connections — that is not reducible to the histological laminar profile as measured by BigBrain staining intensity. The computational consequences of connectivity topology (recurrent vs. feedforward wiring) appear to be the relevant architectural feature for rotational dynamics, rather than the morphological laminar structure per se.

### Convergence with Mesulam classification

The BigBrain null result converges with the Mesulam cytoarchitectural classification analysis reported in the main text. Mesulam type (idiotypic → unimodal → heteromodal → paralimbic) showed no relationship with ρ (ρₛ = −0.027, n.s.; mean ρ = 0.601 ± 0.004 across all four types). Together, these results demonstrate that the ρ gradient is not organized along the classical laminar differentiation axis that distinguishes primary sensory cortex from limbic cortex, despite the strong dorsoventral spatial structure shared by ρ and cytoarchitectural type. The architectural feature that predicts ρ — the dual-origin structural covariance gradient — operates at a different level of organization: it reflects genetically determined, phylogenetically conserved patterns of inter-regional connectivity that produce distinct local circuit dynamics (rotational vs. integrative) independent of the local laminar composition.

## References

Amunts, K., et al. (2013). BigBrain: An ultrahigh-resolution 3D human brain model. *Science,* 340, 1472–1475.

Glasser, M. F., et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature,* 536, 171–178.

Larivière, S., et al. (2021). The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. *Nature Methods,* 18, 698–700.

Paquola, C., et al. (2019). Microstructural and functional gradients are increasingly dissociated in transmodal cortices. *PLOS Biology,* 17, e3000284.
