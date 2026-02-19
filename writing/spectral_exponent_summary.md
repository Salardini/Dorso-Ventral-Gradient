# Spectral Exponent Analysis Summary

## The Question
The paper reports ρ vs spectral exponent r = −0.90. Is ρ just a proxy for the spectral exponent, or does the dual-origin G2 gradient capture unique variance?

## The Answer: G2 gets STRONGER after controlling for spectral exponent

| Predictor | r\|z | r\|z+SE | p\|z+SE |
|-----------|------|---------|---------|
| **SCov G2** | **−0.213** | **−0.253** | **3.1×10⁻⁷** |
| **Genetic G2** | **−0.198** | **−0.235** | **2.0×10⁻⁶** |
| FC Gradient 1 | −0.110 | −0.242 | *** |
| FC clustering | +0.051 | +0.195 | *** |
| FC participation | +0.060 | +0.109 | * |
| SCov G1 | +0.024 | +0.068 | n.s. |
| FC Gradient 2 | −0.022 | −0.080 | n.s. |
| Mesulam | −0.038 | +0.048 | n.s. |

## Variance Explained (R²)

| Model | R² | ΔR² vs previous |
|-------|-----|-----------------|
| z alone | 0.523 | — |
| Spectral exponent alone | 0.795 | — |
| z + spectral exponent | 0.868 | — |
| z + G2 | 0.540 | +0.018 vs z |
| z + spec_exp + G2 | 0.877 | +0.009 vs z+SE |
| z + spec_exp + G2 + GenG2 | 0.877 | +0.000 |

## Key Interpretations

1. **Spectral exponent dominates** (R² = 0.795 alone), but ρ is NOT redundant with it — z adds 7.3% beyond SE, and G2 adds 0.9% beyond both.

2. **G2 effect strengthens** from r = −0.213 (z only) to r = −0.253 (z + SE). Removing spectral exponent variance sharpens the developmental signal.

3. **FC Gradient 1 emerges** after SE control (r = −0.242***). The sensorimotor→default hierarchy predicts ρ once broadband power is removed.

4. **FC clustering becomes significant positive** (r = +0.195***). More locally clustered regions → higher ρ. Consistent with recurrence hypothesis.

5. **Mesulam still null**. Cytoarchitectural type doesn't predict ρ in any model.

## For Reviewers
The dual-origin structural gradient captures variance in ρ that is independent of spatial position, spectral exponent, and functional connectivity hierarchy. The spectral exponent and ρ co-vary because both reflect local circuit architecture (recurrence → oscillatory dynamics + steep 1/f slope), but G2 predicts the residual structure that neither z nor SE explains.

## Files
- `rho_master.csv` — full dataset with all predictors + spectral features + fMRI measures (75 columns)
- Previous: `rho_with_bigbrain.csv`, `rho_with_valk.csv`
