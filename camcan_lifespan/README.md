# CamCAN lifespan extension of the dorsoventral ρ-gradient

Extends the MEG ρ (rotational dynamics) / τ (intrinsic timescale) framework to the
**CamCAN** healthy-ageing cohort (resting MEG, n≈646, ages 18–88, Schaefer-400).
Raw data and per-subject derivatives are **not** in the repo (≈1.3 TB); this folder
holds the analysis scripts, figures, and small result tables.

## Key findings

### 1. Lifespan trajectory of the DV ρ-gradient (`age_state_analysis.py`, `age_state_v2.py`)
- The published −0.73 rest gradient **replicates in age-matched young adults**
  (passive task, 18–30: ρ_s = −0.73, spatial map r = 0.90), but the full-cohort
  estimate is ≈0 — the gradient was hidden by **age-averaging**.
- The DV ρ-gradient **erodes non-linearly with age** (quadratic ≫ linear, ΔAIC≈21;
  age p<1e-24): steep collapse through young adulthood, plateau by mid-life; rest
  **inverts** (~+0.1) by age ~43, passive stays weakly negative across the lifespan.
- **State** (rest vs passive) and **sex** (male stronger) are **additive offsets**
  with **no interaction** with age or each other. τ-DV is age-INVARIANT (~−0.8) —
  ageing targets ρ (dissipation/rotation), not the timescale hierarchy.

### 2. Brain–cognition mapping (`parcel_cognition.py`, `network_cognition.py`)
Per-parcel / per-Yeo-network partial correlations with a cognitive battery
(age+sex adjusted, BH-FDR). Whole-brain gradient summaries are **avoided** — they
spatially average opposite-signed couplings and produce a paradoxical direction.
- **Fluency ↔ frontoparietal Control network ρ (+0.14)** — executive task, executive network.
- **Vocabulary (Spot-the-Word) ↔ Visual/Limbic ρ (−0.14/−0.13)** — reading/ventral stream.
- Effects are **within-age (suppression-type)** and modest; τ shows **no** cognitive map.
- **Episodic memory does not localise** — its substrate (mesial temporal) is absent
  from the cortical Schaefer atlas, and precuneus/PCC are poorly seen by MEG. Memory
  is **excluded** from claims, not reported as null.

### 3. Cross-modal concordance with fMRI (`fmri_concordance.py`)
ρ cannot be measured on fMRI, but the MEG ρ/τ maps concord spatially with fMRI
slow-dynamics variables (spin-test significant): fMRI timescale, fALFF, spectral
exponent, fMRI-ρ. Notably **MEG-τ and fMRI-τ are spatially ANTI-correlated (−0.28)**
— the millisecond and second timescale hierarchies run opposite ways across cortex.
(Parcels aligned by optimal centroid assignment; naive row-order gives x-corr 0.755
and flips signs.)

## Scripts
| script | purpose |
|---|---|
| `camcan_ingest.py` | per-subject ρ/τ extraction (delay-embedded VAR(1) ρ, ACF-integral τ) |
| `assess.py` | evaluate a run's DV gradient vs the published MOUS map |
| `age_state_analysis.py` | lifespan curve + age×state×sex mixed model |
| `age_state_v2.py` | bootstrap-CI curves, quadratic age model, sex-split |
| `parcel_cognition.py` | per-parcel brain–cognition partial-correlation maps |
| `network_cognition.py` | Yeo-7 network × cognitive-domain table + heatmap |
| `fmri_concordance.py` | MEG ρ/τ vs fMRI variables, spin-tested |

## Caveats
Within-age / suppression-type cognitive effects; one parcellation and battery;
cross-modal fMRI comparison is spatial-correlational. See commit history for the
several eyeball-claims that formal tests corrected (age vs state, buffering vs
additive, sex×age, sqrt-combination, the parcel-alignment bug).
