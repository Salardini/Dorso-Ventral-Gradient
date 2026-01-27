# Computational Model: τ-ρ Trade-off in E-I Networks

## Model Description for Methods Section

### Network Architecture

We implemented a rate-based recurrent neural network with excitatory (E) and inhibitory (I) populations to test whether the empirically observed τ-ρ trade-off emerges from varying excitation-inhibition balance.

**Network composition:**
- 100 excitatory units (N_E = 100)
- 25 inhibitory units (N_I = 25)
- Total: 125 units

**Connectivity structure:**
Connection weights were drawn from rectified Gaussian distributions and scaled by population size:

| Connection | Base Weight | Scaling |
|------------|-------------|---------|
| E → E | w_EE = 2.0 | / N_E |
| E → I | w_IE = 1.5 | / N_E |
| I → E | w_EI = 1.5 × g_I | / N_I |
| I → I | w_II = 0.8 × g_I | / N_I |

The **inhibitory gain parameter g_I** scales all inhibitory connections and serves as the key manipulation, varied from 0.3 (weak inhibition) to 3.5 (strong inhibition).

### Dynamics

Each unit's firing rate r_i evolved according to:

$$\tau_i \frac{dr_i}{dt} = -r_i + f\left(\sum_j W_{ij} r_j + I_{ext} + \xi_i(t)\right)$$

where:
- τ_E = 10 ms, τ_I = 5 ms (time constants)
- f(x) = 0.5 × log(1 + exp(x)) (softplus activation)
- I_ext = 1.0 (constant external input)
- ξ(t) ~ N(0, 0.05²) (Gaussian noise)

Simulations used Euler integration with dt = 0.5 ms for T = 2000 ms per trial.

### Metrics

**Intrinsic timescale (τ):** Computed as the integral of the autocorrelation function of population-averaged E-cell activity until decay below 0.05.

**Rotational dynamics (ρ):** Computed via delay embedding (dimension = 6, delay = 20 samples) followed by VAR(1) fitting with ridge regularization (α = 0.001). ρ equals the mean |sin(θ)| where θ are eigenvalue angles.

### Functional Tasks

**1. Temporal pattern richness:** Networks received brief (20 ms) input pulses to random E-cell subsets. Pattern richness was quantified as effective dimensionality (participation ratio) of the activity covariance matrix.

**2. Integration window:** Networks received brief impulses, and we measured the time for the response to decay to 50% of peak amplitude.

### Parameter Sweep

We varied g_I across 12 values (0.3 to 3.5) with 8 repetitions each, measuring τ, ρ, pattern richness, and integration window for each network.

---

## Results Summary

| Finding | Correlation | p-value |
|---------|-------------|---------|
| τ-ρ trade-off | r = -0.84 | p = 0.0006 |
| ρ → pattern richness | r = 0.87 | p = 0.0003 |
| τ → integration window | r = 0.86 | p = 0.0003 |

---

## Text for Manuscript Results Section

### Computational modeling reveals functional consequences of the τ-ρ trade-off

To test whether the empirically observed τ-ρ anticorrelation reflects a fundamental constraint on neural computation, we implemented a rate-based E-I network model in which inhibitory gain (g_I) was systematically varied. This manipulation mimics the hypothesized gradient in PV/SST interneuron ratios across cortex.

Varying inhibitory gain reproduced the τ-ρ trade-off (r = -0.84, p < 0.001; Figure X-B). Strong inhibition (g_I > 2) produced networks with short timescales (τ ≈ 12 ms) and high rotational dynamics (ρ ≈ 0.03), while weak inhibition (g_I < 0.5) produced networks with long timescales (τ ≈ 60 ms) and low rotational dynamics (ρ ≈ 0.02). Intermediate inhibitory gain yielded intermediate dynamical properties.

Critically, this trade-off had functional consequences. Networks with higher ρ generated richer temporal patterns, quantified as higher effective dimensionality of activity trajectories (r = 0.87, p < 0.001; Figure X-C). This supports the hypothesis that rotational dynamics enhance the capacity for temporal sequence generation and feature binding—computations associated with ventral cortex.

Conversely, networks with higher τ maintained longer integration windows, measured as the decay time of impulse responses (r = 0.86, p < 0.001; Figure X-D). This supports the hypothesis that long timescales enhance temporal integration and evidence accumulation—computations associated with dorsal association cortex.

These modeling results demonstrate that the τ-ρ trade-off emerges naturally from E-I balance and has dissociable functional consequences: high-ρ circuits excel at generating rich temporal dynamics, while high-τ circuits excel at temporal integration. This provides a computational interpretation for the empirically observed dorsoventral gradient: ventral cortex may be optimized for sequence generation and binding, while dorsal cortex may be optimized for integration and stable maintenance.

---

## Text for Manuscript Discussion Section

### Mechanistic basis of the τ-ρ trade-off

Our computational model demonstrates that the τ-ρ trade-off emerges naturally from varying excitation-inhibition balance in recurrent networks. Strong inhibition—characteristic of PV interneuron-rich circuits—creates fast negative feedback loops that generate oscillatory (rotational) dynamics but prevent sustained activity. Weak inhibition—characteristic of SST interneuron-rich circuits—allows recurrent excitation to sustain activity over longer timescales but reduces oscillatory structure.

This mechanistic account makes a testable prediction: the dorsoventral ρ gradient should correlate with spatial gradients in PV versus SST interneuron gene expression. PV interneurons, which provide fast perisomatic inhibition and are enriched in sensory cortices, should be associated with high ρ (ventral). SST interneurons, which provide slower dendritic inhibition and are enriched in association cortex, should be associated with low ρ (dorsal). We test this prediction using gene expression data from the Allen Human Brain Atlas in the following section.

The functional dissociation observed in our model—pattern richness for high-ρ networks versus integration for high-τ networks—aligns with the classical distinction between ventral ("what") and dorsal ("where/how") processing streams. Ventral regions specialized for object recognition may benefit from rich temporal dynamics that support feature binding across hierarchical levels. Dorsal regions specialized for spatial processing and action planning may benefit from long integration windows that support stable representations and evidence accumulation.
