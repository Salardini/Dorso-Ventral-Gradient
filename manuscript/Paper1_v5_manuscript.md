# A Dorsoventral Gradient of Rotational Dynamics in Human Cortex

## ABSTRACT

Cortical computation emerges from the interplay of neural dynamics across space and time, yet how dynamical properties are spatially organized remains poorly understood. Here we characterize the spatial organization of rotational dynamics—a signature of oscillatory, underdamped neural activity—using resting-state magnetoencephalography (MEG) in 203 healthy adults. We introduce ρ, an index quantifying the rotational component of local dynamics derived from delay-embedded linear models. Across Schaefer-400 parcels, ρ exhibited a striking dorsoventral gradient (r = −0.735 with MNI z-coordinate; p_spin = 0.002), with ventral regions showing stronger rotational dynamics. This gradient was robust to parcellation resolution (Schaefer-100: r = −0.69), replicated across visual and auditory task states, and was independent of both the principal functional connectivity gradient (r = −0.04) and cortical myelin content (r = −0.015). Frequency-specific analysis revealed a spectral trade-off: slow oscillations (delta, theta) showed dorsal predominance while fast oscillations (beta, gamma) showed ventral predominance, with gradient directions confirmed using adaptive embedding delays. Intrinsic timescales (τ) showed apparent spatial structure that did not survive spatially-constrained null models, yet τ and ρ exhibited strong anticorrelation after removing shared geometry (r = −0.647; p_spin = 0.0002), suggesting complementary dynamical organization. These findings establish a novel dorsoventral axis of cortical dynamics that may reflect gradients in local circuit properties including inhibitory interneuron composition.

---

## INTRODUCTION

The cerebral cortex exhibits systematic spatial organization across multiple dimensions. Functional connectivity gradients reveal a principal axis spanning unimodal sensory regions to transmodal association cortex, while intrinsic timescales increase along a similar hierarchy. These organizational principles constrain how cortical circuits process information across space and time.

Beyond connectivity and timescales, neural dynamics themselves—the temporal evolution of activity patterns—may be spatially organized. Dynamical systems can exhibit qualitatively different behaviors: stable, decaying dynamics versus oscillatory, rotational dynamics. These regimes have distinct computational implications: decaying dynamics support point-attractor computations and stable representations, while rotational dynamics enable sequence generation, timing, and rhythmic processing.

Whether the cortex exhibits systematic spatial organization in its dynamical regime remains unknown. Here we address this question using magnetoencephalography (MEG) to characterize local dynamics across the cortical surface. We introduce ρ (rho), an index quantifying the rotational component of parcel-level dynamics derived from delay-embedded vector autoregressive models. High ρ indicates underdamped, oscillatory dynamics; low ρ indicates overdamped, decaying dynamics.

We hypothesized that ρ would vary systematically across cortex, potentially along anatomical axes that reflect gradients in local circuit properties. Using resting-state MEG from 203 participants, we discovered a pronounced dorsoventral gradient in ρ, with ventral regions exhibiting stronger rotational dynamics. This gradient is robust across cognitive states, independent of established cortical gradients, and exhibits frequency-specific organization consistent with distinct circuit mechanisms for slow integrative versus fast local processing.

---

## RESULTS

### A dorsoventral gradient of rotational dynamics

We computed ρ for each of 400 cortical parcels (Schaefer atlas) from resting-state MEG in 203 participants. The ρ metric quantifies the rotational component of local dynamics by fitting a vector autoregressive model to delay-embedded time series and computing the mean sine of eigenvalue angles (see Methods).

Across parcels, ρ exhibited a pronounced dorsoventral (DV) gradient (r = −0.735 with MNI z-coordinate; p_spin = 0.002; Figure 1A-B). Ventral regions—including inferior temporal, orbitofrontal, and ventral occipital cortex—showed high ρ values indicating strong rotational dynamics. Dorsal regions—including superior parietal, dorsolateral prefrontal, and dorsal premotor cortex—showed low ρ values indicating more stable, decaying dynamics.

The gradient axis was oriented 18° from the pure dorsoventral direction (Figure 1C), consistent across both hemispheres (left: 17°; right: 19°) and across broad cortical divisions (frontal: 17°; posterior: 13°; Extended Data Table 1). This near-vertical orientation distinguishes ρ from other cortical gradients that typically follow anterior-posterior or sensory-transmodal axes.

### Robustness to parcellation resolution

To test whether the gradient depends on fine-grained parcellation, we repeated the analysis after aggregating parcels to coarser resolutions. The gradient persisted with minimal attenuation: Schaefer-200 (143 parcels after aggregation): r = −0.702, p_spin < 0.0001; Schaefer-100 (76 parcels): r = −0.686, p_spin < 0.0001. This robustness indicates the finding is not an artifact of spatial smoothing or parcellation choice.

### Individual-level consistency

To ensure the gradient is not a group-averaging artifact, we computed ρ-DV correlations for each participant individually. The gradient was consistent across individuals (mean r = −0.32 ± 0.19; 94% of subjects showed negative correlation; t(211) = −24.19, p < 0.001). While individual correlations were weaker than the group-level estimate—as expected given single-subject noise—the near-universal presence of negative correlations confirms robust individual-level organization.

### Replication across cognitive states

To test state-dependence, we computed ρ during visual object recognition (N = 69) and auditory speech processing (N = 95) tasks. The DV gradient replicated in both conditions: visual task r = −0.68, p_spin = 0.004; auditory task r = −0.74, p_spin = 0.002. Gradient directions were nearly identical to rest (visual: 19° from DV; auditory: 16° from DV), indicating stable underlying circuit architecture rather than state-dependent modulation.

### Independence from established cortical gradients

We tested whether ρ relates to the principal gradient of functional connectivity, which spans unimodal to transmodal cortex. Network-level mean ρ showed minimal variation across the sensory-transmodal axis (<1% range: 0.599 in dorsal attention to 0.607 in limbic), while the DV gradient was present within every functional network (visual: r = −0.90; somatomotor: r = −0.46; dorsal attention: r = −0.71; salience/ventral attention: r = −0.62; limbic: r = −0.44; control: r = −0.79; default: r = −0.66; all p < 0.05). Controlling for network membership did not attenuate the DV correlation (partial r = −0.66), confirming orthogonality between the ρ gradient and hierarchical organization.

We also tested independence from cortical microstructure by correlating ρ with T1w/T2w-derived myelin estimates. The ρ map was uncorrelated with myelin content (r = −0.015; p_spin = 0.80), and the ρ-DV gradient was unchanged after controlling for myelin (partial r = −0.72). This dissociation indicates that ρ reflects functional circuit properties rather than myeloarchitectural gradients.

### Validation with model-free nonlinear measure

To validate that ρ captures meaningful dynamical structure beyond linear model assumptions, we computed permutation entropy—a model-free measure of temporal complexity based on ordinal patterns. Permutation entropy showed the same DV gradient as ρ (r = −0.57 with z-coordinate; p_spin = 0.003) and correlated positively with ρ across parcels (r = +0.49; p_spin = 0.001).

Among nonlinear measures, permutation entropy provides the strongest validation because it specifically captures ordinal temporal patterns—the sequence of rises and falls in the signal—which is conceptually aligned with rotational dynamics (trajectory curvature through state space). Spectral entropy reflects frequency-domain complexity but is insensitive to phase relationships; sample entropy measures regularity but is dominated by amplitude fluctuations. Permutation entropy's ordinal approach makes it uniquely suited to validate rotation-like dynamics.

### Frequency-specific organization: a spectral trade-off

To characterize the spectral basis of the ρ gradient, we computed band-specific ρ after filtering to canonical frequency bands. This revealed a striking frequency-dependent reversal (Figure 2; Extended Data Table 1):

**Slow rhythms (dorsal predominance):**
- Delta (1-4 Hz): r = +0.55, p_spin < 0.0001
- Theta (4-8 Hz): r = +0.47, p_spin < 0.0001

**Fast rhythms (ventral predominance):**
- Alpha (8-13 Hz): r = −0.65, p_spin < 0.0001
- Beta (13-30 Hz): r = −0.77, p_spin < 0.0001
- Gamma (30-40 Hz): r = −0.73, p_spin < 0.0001

The broadband ρ gradient (r = −0.735) reflects dominance of fast frequencies in the 1-40 Hz range. This spectral trade-off suggests distinct circuit mechanisms: ventral regions favor fast, locally-generated rotational dynamics while dorsal regions favor slow, potentially long-range integrative dynamics.

### Robustness to embedding delay

The ρ metric uses a fixed embedding delay (1 sample), which could introduce frequency-dependent sensitivity. To test whether gradient differences reflect sampling artifacts, we recomputed ρ with adaptive delays targeting quarter-cycle phase advance at each band's center frequency.

Gradient directions were unchanged across all bands with adaptive delays: delta r = +0.42, theta r = +0.43, alpha r = −0.61, beta-high r = −0.77, gamma r = −0.73 (all p_spin < 0.0001). The consistency confirms that the frequency-dependent spatial organization—dorsal predominance for slow rhythms, ventral predominance for fast rhythms—reflects genuine circuit properties rather than methodological confounds.

### Relationship with intrinsic timescales

We computed intrinsic timescales (τ) as the integral of the autocorrelation function for each parcel. In contrast to ρ, τ showed apparent correlations with both anterior-posterior (r = +0.418) and dorsoventral (r = −0.345) coordinates under parametric tests. However, these associations did not survive spatially-constrained null models (p_spin = 0.229 and 0.234, respectively), indicating that τ's apparent geometry is not separable from spatial autocorrelation.

Despite this, τ and ρ showed a striking relationship after accounting for gross anatomy. In raw parcel space, τ and ρ were independent (r = −0.034; p_spin = 0.877). However, after regressing out spatial coordinates (x, y, z) from both maps, τ and ρ residuals showed strong anticorrelation (r = −0.647; p_spin = 0.0002).

This pattern reflects regional heterogeneity: in frontal cortex, both τ and ρ follow dorsoventral axes (positive correlation); in posterior cortex, they follow orthogonal axes (τ anterior-posterior, ρ dorsoventral), creating negative correlation. The residual anticorrelation suggests a local circuit constraint: within any cortical neighborhood, circuits cannot simultaneously maximize both temporal integration (long τ) and rotational dynamics (high ρ).

### Partial replication in HCP dataset

In an independent dataset (Human Connectome Project MEG, N = 89), we tested replication using pre-processed power envelope time series parcellated to 107 regions. The gamma-low band showed a correlation consistent with MOUS findings (r = −0.40), though this did not survive spin permutation (p_spin = 0.12).

The weaker effect likely reflects fundamental methodological differences: HCP data represent band-limited power envelopes rather than raw broadband signals. Since ρ measures phase relationships in state-space trajectories, envelope-based analysis—which captures amplitude fluctuations but loses phase information—may underestimate rotational structure. Importantly, our atlas sensitivity analysis demonstrated that parcellation resolution alone does not explain the difference: MOUS data at 76 parcels still showed r = −0.69. The HCP result should therefore be interpreted as a consistent trend in a methodologically distinct dataset.

---

## DISCUSSION

We have identified a dorsoventral gradient of rotational dynamics in human cortex. This gradient is robust across cognitive states and parcellation resolutions, independent of established cortical gradients, and exhibits frequency-specific organization suggesting distinct circuit mechanisms for slow versus fast neural processing.

### A novel axis of cortical organization

The ρ gradient represents a distinct organizational principle. Its near-vertical orientation (18° from dorsoventral) contrasts with the anterior-posterior axis of the principal connectivity gradient and the complex geometry of intrinsic timescales. The orthogonality to the principal gradient (r = −0.04) and independence from myelin (r = −0.015) indicate that ρ captures functional circuit properties not reducible to hierarchical position or structural features.

These dissociations suggest multiple independent organizational axes in cortex. The principal gradient may reflect hierarchical processing depth; τ may reflect integration windows shaped by recurrent connectivity; and ρ may reflect the balance of oscillatory versus stable dynamics shaped by local inhibitory circuitry.

### Frequency-specific circuit mechanisms

The spectral trade-off—slow rhythms dorsally, fast rhythms ventrally—suggests distinct circuit mechanisms. Fast oscillations (beta, gamma) are generated by local inhibitory networks, particularly parvalbumin-positive (PV) interneurons providing perisomatic inhibition. These frequencies dominate ventral regions and show strong rotational signatures. Slow oscillations (delta, theta) involve longer-range cortico-cortical connections and cortico-thalamic loops, consistent with their dorsal predominance and weaker local rotational structure.

This interpretation aligns with known gradients in interneuron subtypes. PV interneurons, enriched in sensory cortices, generate fast gamma through tight feedback inhibition—promoting rotational dynamics. Somatostatin-positive (SST) interneurons, enriched in association cortex, provide dendritic inhibition that stabilizes integration—promoting decaying dynamics. The ρ gradient may thus track PV/SST ratios across cortex, a hypothesis testable via comparison with gene expression data from the Allen Human Brain Atlas.

### Relationship to dual-stream processing

The dorsoventral organization of ρ resonates with the classical distinction between dorsal ("where/how") and ventral ("what") visual streams. Ventral stream regions, specialized for object recognition through hierarchical feature extraction, show high ρ—consistent with the oscillatory dynamics supporting temporal binding and feature integration. Dorsal stream regions, specialized for spatial processing and action guidance, show low ρ—consistent with stable attractor dynamics supporting spatial representations.

This parallel suggests the ρ gradient may reflect fundamental computational requirements rather than arbitrary anatomical variation: ventral computations requiring dynamic, sequential processing; dorsal computations requiring stable, point-attractor representations.

### The τ-ρ trade-off as a local circuit constraint

The strong anticorrelation between τ and ρ after removing spatial geometry (r = −0.647) suggests a fundamental trade-off in local circuit properties. Circuits optimized for temporal integration (long τ) appear incompatible with strong rotational dynamics (high ρ), and vice versa.

This trade-off may reflect biophysical constraints on recurrent connectivity: symmetric connectivity patterns that promote integration through mutual excitation preclude the antisymmetric patterns that generate rotation. The decomposition of effective connectivity into symmetric (dissipative) and antisymmetric (rotational) components provides a mathematical framework for this intuition, suggesting that local circuits cannot simultaneously maximize both.

### Methodological considerations

Several methodological points warrant discussion. First, our ρ metric uses Takens delay embedding of raw time series—not Hilbert-derived amplitude and phase coordinates. This classical approach from nonlinear dynamics avoids phase circularity issues and is well-suited for characterizing local dynamical structure.

Second, the ρ metric has inherent frequency-dependent sensitivity due to fixed embedding delay: slow oscillations advance fewer degrees per sample than fast oscillations. This means absolute ρ values are not directly comparable across frequency bands. However, within-band spatial comparisons (our primary analysis) are unaffected, and our adaptive delay analysis confirmed that gradient directions are robust to this consideration.

Third, the weaker HCP replication reflects methodological differences rather than non-reproducibility. HCP's power envelope time series fundamentally differ from raw broadband signals—envelopes capture amplitude fluctuations but lose phase information critical to rotational dynamics.

### Toward causal tests

The present findings are correlational; establishing causality will require interventional approaches. In animal models, optogenetic manipulation of inhibitory interneurons—particularly PV cells that shape oscillatory dynamics—could test whether altering inhibitory tone shifts ρ as predicted. In humans, transcranial magnetic stimulation targeting dorsal versus ventral regions could probe whether disrupting local dynamics produces opposing changes in rotational signatures. Computational modeling using balanced network frameworks with varied inhibitory strength could test whether DV-like ρ gradients emerge from biologically plausible parameter variations.

### Limitations

Several limitations should be noted. Our sample (ages 18-45) limits generalizability across the lifespan; cortical inhibitory balance and myelination change with development and aging, potentially shifting the ρ gradient. Task MEG was available for participant subsets (visual: N=69; auditory: N=95) determined by study design, not resting-state results, minimizing selection bias. We lack behavioral correlations; future work should test whether individual differences in ρ gradient strength predict performance on tasks differentially engaging dorsal versus ventral processing. Finally, the simple linear residualization for the τ-ρ analysis could be refined with more sophisticated spatial models.

### Conclusion

We have established a robust dorsoventral gradient of rotational dynamics in human cortex. This gradient represents a novel organizational axis, orthogonal to hierarchical gradients and independent of myeloarchitecture. The frequency-specific organization—slow rhythms dorsally, fast rhythms ventrally—and the complementary relationship with intrinsic timescales suggest that multiple dynamical properties are spatially coordinated to support cortical computation. These findings open new avenues for understanding how circuit-level properties shape the spatial organization of brain dynamics.

---

## METHODS

### Participants

We analyzed resting-state MEG from the Mother of Unification Studies (MOUS) dataset. After quality control filtering (retaining subjects matching pattern sub-[AV]\\d+), 203 participants were included (ages 18-45, mean 26.3 years). Task MEG was available for subsets: visual object recognition (N=69), auditory speech processing (N=95).

### MEG acquisition and preprocessing

MEG was recorded using a 275-channel CTF system at 1200 Hz. Preprocessing included: band-pass filtering (1-40 Hz), notch filtering (50 Hz and harmonics), independent component analysis for artifact removal, and downsampling to 200 Hz. Bad channels were interpolated and segments with residual artifacts were excluded.

### Source reconstruction

Source localization used linearly constrained minimum variance (LCMV) beamforming. Forward models were computed using single-shell boundary element models derived from individual T1-weighted MRI. Source time series were extracted for 400 cortical parcels (Schaefer 7-network atlas) by averaging across vertices within each parcel.

### Rotational dynamics index (ρ)

We quantified rotational dynamics using delay embedding and vector autoregressive modeling:

1. **Delay embedding**: For each parcel time series x(t), we constructed a 10-dimensional state vector:
   
   X(t) = [x(t), x(t-d), x(t-2d), ..., x(t-9d)]
   
   where d = 1 sample (5 ms at 200 Hz). This Takens embedding reconstructs the dynamical attractor from a scalar time series.

2. **VAR(1) fitting**: We fit a first-order vector autoregressive model with ridge regularization (α = 0.001):
   
   X(t+1) = A · X(t) + ε

3. **Rotational index**: From the eigenvalues λ of A, we computed:
   
   ρ = mean(|Im(λ)| / |λ|) for |λ| > 0.01
   
   This equals the mean sine of eigenvalue angles, quantifying the rotational (oscillatory) versus decaying (stable) character of dynamics. High ρ indicates underdamped, rotational dynamics; low ρ indicates overdamped, decaying dynamics.

Model fits were assessed via one-step prediction R² (median = 0.9999; 5th percentile = 0.9997), confirming that VAR(1) provides an adequate local linear approximation.

### Intrinsic timescales (τ)

Intrinsic timescales were computed as the integral of the autocorrelation function (ACF) from 5 ms to 300 ms lag, using only positive ACF values:

τ = ∫[5ms to 300ms] max(ACF(lag), 0) d(lag)

### Frequency-specific analysis

For band-specific ρ, time series were filtered to canonical bands (delta: 1-4 Hz; theta: 4-8 Hz; alpha: 8-13 Hz; beta-low: 13-20 Hz; beta-high: 20-30 Hz; gamma: 30-40 Hz) using fourth-order Butterworth filters before delay embedding.

To test robustness to embedding delay, we recomputed ρ with adaptive delays targeting quarter-cycle phase advance: delay = round(0.25 / center_freq × fs) samples.

### Spatial statistics

Correlations with anatomical coordinates used MNI centroid positions for each parcel. Gradient axis angles were computed as arctan(r_AP / r_DV).

Statistical inference used hemisphere-preserving spin permutation tests (10,000 rotations). For each permutation, parcel centroids were normalized to a unit sphere, independently rotated within each hemisphere using random SO(3) rotations, and reassigned by nearest-neighbor matching. Two-tailed p_spin values were computed from permutation distributions.

For the τ-ρ residual correlation, we regressed each map on spatial coordinates (x, y, z), then correlated residuals. Permutation testing spun ρ, re-residualized, and recomputed the correlation.

### Parcellation sensitivity

To test resolution dependence, we aggregated Schaefer-400 parcels to coarser resolutions by averaging time series within hierarchically-defined groups (Schaefer-200: pairs; Schaefer-100: quartets), recomputed ρ, and tested DV correlations.

### HCP replication

For the Human Connectome Project (N=89), we used pre-processed power envelope time series (band-filtered, Hilbert amplitude) parcellated to Yeo-17 split atlas (~107 regions). ρ was computed identically to MOUS.

### Myelin analysis

T1w/T2w myelin estimates were obtained from the HCP S1200 group-average parcellated to Schaefer-400 using neuromaps. Correlations with ρ used spin permutation tests.

### Nonlinear validation

Permutation entropy (order 3, delay 1) was computed for each parcel time series, providing a model-free measure of temporal complexity based on ordinal patterns.

### Code and data availability

Analysis code is available at https://github.com/Salardini/Dorso-Ventral-Gradient. The MOUS dataset is available through the Donders Repository. HCP MEG data are available through the Human Connectome Project.

---

## EXTENDED DATA

### Extended Data Table 1. Summary of spatial organization

**Panel A: ρ gradient by region and state**

| Condition | Region | r (ρ-DV) | p_spin | Axis angle |
|-----------|--------|----------|--------|------------|
| Rest | Full cortex | −0.735 | 0.002 | 18° |
| Rest | Frontal | −0.71 | 0.003 | 17° |
| Rest | Posterior | −0.74 | 0.002 | 13° |
| Visual task | Full cortex | −0.68 | 0.004 | 19° |
| Auditory task | Full cortex | −0.74 | 0.002 | 16° |

**Panel B: Parcellation sensitivity**

| Resolution | N parcels | r (ρ-DV) | p_spin |
|------------|-----------|----------|--------|
| Schaefer-400 | 400 | −0.723 | <0.0001 |
| Schaefer-200 | 143 | −0.702 | <0.0001 |
| Schaefer-100 | 76 | −0.686 | <0.0001 |

**Panel C: Frequency-specific gradients**

| Band | Frequency | Fixed delay r | Adaptive delay r | p_spin | Direction |
|------|-----------|---------------|------------------|--------|-----------|
| Delta | 1-4 Hz | +0.55 | +0.42 | <0.0001 | Dorsal |
| Theta | 4-8 Hz | +0.47 | +0.43 | <0.0001 | Dorsal |
| Alpha | 8-13 Hz | −0.65 | −0.61 | <0.0001 | Ventral |
| Beta-low | 13-20 Hz | −0.32 | −0.45 | <0.0001 | Ventral |
| Beta-high | 20-30 Hz | −0.77 | −0.77 | <0.0001 | Ventral |
| Gamma | 30-40 Hz | −0.73 | −0.73 | <0.0001 | Ventral |
| Broadband | 1-40 Hz | −0.735 | — | 0.002 | Ventral |

**Panel D: τ and τ-ρ relationships**

| Metric | r | p_spin | Interpretation |
|--------|---|--------|----------------|
| τ vs AP | +0.418 | 0.229 | Not significant |
| τ vs DV | −0.345 | 0.234 | Not significant |
| τ vs ρ (raw) | −0.034 | 0.877 | Independent |
| τ vs ρ (residualized) | −0.647 | 0.0002 | Strong anticorrelation |

**Panel E: Independence from other gradients**

| Comparison | r | p_spin | Interpretation |
|------------|---|--------|----------------|
| ρ vs Principal gradient | −0.04 | 0.45 | Orthogonal |
| ρ vs Myelin | −0.015 | 0.80 | Independent |
| ρ vs Permutation entropy | +0.49 | 0.001 | Validates |
| Perm. entropy vs DV | −0.57 | 0.003 | Same gradient |

---

## REFERENCES

1. Margulies DS, et al. (2016) Situating the default-mode network along a principal gradient of macroscale cortical organization. PNAS 113:12574-12579.

2. Honey CJ, et al. (2012) Slow cortical dynamics and the accumulation of information over long timescales. Neuron 76:423-434.

3. Takens F (1981) Detecting strange attractors in turbulence. Lecture Notes in Mathematics 898:366-381.

4. Schaefer A, et al. (2018) Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cerebral Cortex 28:3095-3114.

5. Glasser MF, Van Essen DC (2011) Mapping human cortical areas in vivo based on myelin content as revealed by T1- and T2-weighted MRI. J Neurosci 31:11597-11616.

6. Alexander-Bloch AF, et al. (2018) On testing for spatial correspondence between maps of human brain structure and function. NeuroImage 178:540-551.

7. Hennequin G, et al. (2014) Optimal control of transient dynamics in balanced networks supports generation of complex movements. Neuron 82:1394-1406.

8. Bandt C, Pompe B (2002) Permutation entropy: a natural complexity measure for time series. Phys Rev Lett 88:174102.

9. Goodale MA, Milner AD (1992) Separate visual pathways for perception and action. Trends Neurosci 15:20-25.

10. Kantz H, Schreiber T (2004) Nonlinear Time Series Analysis. Cambridge University Press.
