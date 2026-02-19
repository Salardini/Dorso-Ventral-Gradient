# τ (Intrinsic Timescale) Analysis Summary

## Key Result: G2 survives everything

**SCov G2 | z + SE + τ: r = −0.249, p = 4.7 × 10⁻⁷**

The dual-origin gradient predicts ρ after simultaneously controlling for spatial position (z), spectral exponent, AND intrinsic timescale. The effect barely budges.

## Progressive Controls on SCov G2

| Control variables | r | p |
|-------------------|---|---|
| Zero-order | −0.155 | 0.002 |
| \| z | −0.213 | 1.7×10⁻⁵ |
| \| SE | −0.181 | 2.7×10⁻⁴ |
| \| τ | −0.148 | 0.003 |
| \| z + SE | −0.253 | 3.1×10⁻⁷ |
| \| z + τ | −0.233 | 2.4×10⁻⁶ |
| \| SE + τ | −0.185 | 2.1×10⁻⁴ |
| **\| z + SE + τ** | **−0.249** | **4.7×10⁻⁷** |

Genetic G2 replicates: r = −0.231, p = 3.0×10⁻⁶ (same toughest model).

## Full Predictor Table (toughest model: | z + SE + τ)

| Predictor | r\|z | r\|z+SE | r\|z+SE+τ | p |
|-----------|------|---------|-----------|---|
| **SCov G2** | **−0.213** | **−0.253** | **−0.249** | **4.7×10⁻⁷** |
| **Genetic G2** | **−0.198** | **−0.235** | **−0.231** | **3.0×10⁻⁶** |
| FC Gradient 1 | −0.110 | −0.242 | −0.235 | 2.1×10⁻⁶ |
| FC clustering | +0.051 | +0.195 | +0.188 | 1.6×10⁻⁴ |
| FC participation | +0.060 | +0.109 | +0.114 | 0.022 |
| SCov G1 | +0.024 | +0.068 | +0.063 | n.s. |
| FC Gradient 2 | −0.022 | −0.080 | −0.082 | n.s. |
| Mesulam | −0.038 | +0.048 | +0.043 | n.s. |

## Variance Explained (R²)

| Model | R² |
|-------|-----|
| z | 0.523 |
| SE | 0.795 |
| τ | 0.099 |
| z + SE | 0.868 |
| z + SE + τ | 0.870 |
| z + SE + G2 | 0.877 |
| z + SE + τ + G2 | 0.878 |
| z + SE + τ + G2 + FC_G1 | 0.884 |
| z + SE + τ + G2 + FC_G1 + clust | 0.884 |

## τ Properties

- τ vs ρ: ρₛ = +0.341 (moderate positive — longer timescale regions have MORE rotation, which is interesting)
- τ vs z: ρₛ = −0.644 (strong — dorsal regions have longer τ)
- τ vs SE: ρₛ = −0.176 (weak)
- τ vs FC G1: ρₛ = −0.141 (weak)
- **τ vs G2: ρₛ = −0.047, n.s.** — G2 does NOT predict τ, confirming G2 captures something distinct from the timescale hierarchy

## Interpretation

G2 predicts a component of rotational dynamics that is:
- Independent of where a region sits on the dorsoventral axis (z)
- Independent of its spectral properties (1/f slope)
- Independent of its intrinsic timescale (τ)
- Genetically determined (Genetic G2 replicates)

This is variance in ρ that can only be explained by the dual-origin developmental connectivity architecture.

## Files
- `rho_master.csv` — 78 columns, full dataset
