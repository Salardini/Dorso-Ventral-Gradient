# Structural Connectivity Analysis: Methods and Results

## Methods

### HCP structural connectivity

Group-average structural connectivity was obtained from the Human Connectome Project (HCP) via the ENIGMA Toolbox (Larivière et al., 2021), parcellated to the Schaefer 400 atlas. The structural connectivity (SC) matrix represents log-transformed streamline counts from deterministic tractography of diffusion-weighted MRI data, averaged across HCP participants. The matrix was 400 × 400, symmetric, and sparse (6.2% non-zero entries). Parcel labels were reindexed to match our analysis order using exact string matching of Schaefer 400 parcel names.

### Structural connectivity metrics

From the SC matrix, we computed the following per-parcel metrics:

- **Node strength**: Sum of all connection weights (total streamline density per parcel).
- **Node degree**: Number of non-zero connections.
- **Mean connection distance**: Weighted average Euclidean distance (mm) to connected parcels, using MNI centroid coordinates.
- **Short-range strength**: Total connection weight to parcels within 50 mm Euclidean distance.
- **Long-range strength**: Total connection weight to parcels ≥50 mm Euclidean distance.
- **Short/long range ratio**: Ratio of short-range to long-range connection strength, indexing the predominance of local vs. distant connectivity.
- **Weighted clustering coefficient**: Computed using the Onnela et al. (2005) formulation (cube root of weights), measuring local circuit triangulation.
- **Participation coefficient**: Diversity of cross-network connectivity (Shannon entropy-based; Newman, 2006), computed with respect to Yeo 7-network assignments.
- **Within-network fraction**: Proportion of total SC strength directed to same-network parcels.

### Statistical analysis

SC metrics were tested against ρ using the same framework as all other predictors: zero-order Spearman correlations, partial correlations controlling for z, and partial correlations controlling for z + spectral exponent + τ. To assess mediation, we tested whether G2 survives controlling for SC metrics and vice versa.

## Results

### Structural connectivity predicts ρ

Several SC metrics showed significant associations with ρ (Table 1). The short/long range ratio — indexing the balance of local vs. distant white matter connections — showed the strongest zero-order correlation (ρₛ = −0.273, p < 0.001), indicating that regions with more locally concentrated structural connectivity exhibit stronger rotational dynamics. This association survived z-correction (r|z = −0.158, p < 0.01) and the full control battery (r|z+SE+τ = −0.115, p < 0.05).

Mean connection distance was also significant (ρₛ = +0.359, p < 0.001; r|z+SE+τ = +0.112, p < 0.05): regions with shorter average connection distances show more rotation. After controlling for z, SE, and τ, SC degree (r = +0.232, p < 0.001), node strength (r = +0.185, p < 0.001), weighted clustering (r = −0.174, p < 0.001), within-network fraction (r = −0.160, p < 0.01), long-range strength (r = +0.144, p < 0.01), and participation coefficient (r = +0.147, p < 0.01) all emerged as significant, suggesting that the spectral exponent had been masking structural connectivity effects.

**Table 1. Structural connectivity metrics vs ρ (n = 400 parcels)**

| Metric | ρₛ | r|z | r|z+SE+τ |
|---|---|---|---|
| SC short/long ratio | −0.273*** | −0.158** | −0.115* |
| SC mean distance | +0.359*** | +0.179*** | +0.112* |
| SC short-range strength | −0.192*** | −0.005 | +0.098 |
| SC long-range strength | +0.228*** | +0.154** | +0.144** |
| SC degree | −0.065 | +0.050 | +0.232*** |
| SC node strength | −0.015 | +0.100* | +0.185*** |
| SC weighted clustering | +0.062 | −0.136** | −0.174*** |
| SC participation coeff. | +0.030 | +0.079 | +0.147** |
| SC within-network frac. | −0.065 | −0.066 | −0.160** |

\* p < 0.05; \*\* p < 0.01; \*\*\* p < 0.001

### SC tracks the dual-origin gradient

The structural covariance gradient G2 was significantly correlated with several SC metrics: short/long range ratio (ρₛ = +0.147, p = 0.003), long-range strength (ρₛ = −0.161, p = 0.001), mean connection distance (ρₛ = −0.141, p = 0.005), and degree (ρₛ = −0.146, p = 0.003). Regions at the paleocortical pole of G2 (negative G2 values) tend to have higher short/long ratios — that is, more locally concentrated connectivity — while archicortical-pole regions have longer-range, more distributed connections. This pattern is consistent with the known wiring properties of paleocortical and archicortical derivatives (Pandya et al., 2015).

### G2 mediates SC effects on ρ

To determine whether the SC–ρ relationship is mediated by the dual-origin gradient, we performed mutual adjustment analyses. After controlling for z, SE, τ, AND the short/long range ratio, G2 remained a robust predictor of ρ (r = −0.238, p = 1.4 × 10⁻⁶). Conversely, after controlling for z, SE, τ, AND G2, the short/long ratio was reduced to non-significance (r = −0.089, p = 0.074). Similarly, SC short-range strength was non-significant after G2 control (r = +0.085, p = 0.090). This pattern indicates that G2 captures the variance in ρ that SC metrics index, but also captures additional developmental/genetic information beyond what diffusion tractography reveals.

In terms of incremental R², SC metrics added modestly beyond the base model (z + SE + τ; R² = 0.870): SC short-range strength added ΔR² = 0.001, SC degree ΔR² = 0.003. G2 alone added ΔR² = 0.008. Adding both G2 and SC degree yielded ΔR² = 0.010, only marginally more than G2 alone. This confirms that SC connectivity range partially mediates the G2–ρ relationship but does not fully account for it.

## References

Larivière, S., et al. (2021). The ENIGMA Toolbox: multiscale neural contextualization of multisite neuroimaging datasets. *Nature Methods,* 18, 698–700.

Onnela, J.-P., et al. (2005). Intensity and coherence of motifs in weighted complex networks. *Physical Review E,* 71, 065103.

Pandya, D. N., et al. (2015). *Cerebral Cortex: Architecture, Connections, and the Dual Origin Concept.* Oxford University Press.
