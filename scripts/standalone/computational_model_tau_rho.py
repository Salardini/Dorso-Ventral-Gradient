#!/usr/bin/env python3
"""
Computational Model: τ-ρ Trade-off in E-I Networks
===================================================

This model demonstrates that the τ-ρ trade-off observed in human cortex
emerges naturally from varying excitation-inhibition balance in recurrent
neural networks, and that this trade-off has functional consequences.

THEORETICAL BACKGROUND
----------------------
The model tests whether a simple manipulation—varying inhibitory gain—can
produce the anticorrelation between intrinsic timescale (τ) and rotational
dynamics (ρ) observed empirically (r = -0.65 after spatial residualization).

Hypothesis: Strong inhibition (mimicking PV interneuron-rich circuits) should
produce fast, oscillatory dynamics (low τ, high ρ), while weak inhibition
(mimicking SST interneuron-rich circuits) should produce slow, stable dynamics
(high τ, low ρ).

MODEL ARCHITECTURE
------------------
- Rate-based recurrent neural network
- N_E = 100 excitatory units, N_I = 25 inhibitory units
- Softplus activation function: f(x) = 0.5 * log(1 + exp(x))
- Time constants: τ_E = 10ms (excitatory), τ_I = 5ms (inhibitory)

Connectivity (scaled by population size):
- E→E: w_EE = 2.0 (strong recurrence, promotes dynamics)
- E→I: w_IE = 1.5
- I→E: w_EI = 1.5 * g_I (KEY PARAMETER: inhibitory gain)
- I→I: w_II = 0.8 * g_I

The inhibitory gain g_I is varied from 0.3 (weak) to 3.5 (strong).

DYNAMICS
--------
Each unit follows:
    τ * dr/dt = -r + f(W @ r + I_ext + noise)

where:
- r is the firing rate vector
- W is the weight matrix
- I_ext is external input (constant baseline = 1.0)
- noise is Gaussian (σ = 0.05)

Simulated with Euler method, dt = 0.5ms.

METRICS
-------
1. Intrinsic timescale (τ):
   - Computed from autocorrelation of population activity
   - τ = ∫ ACF(lag) d(lag) until ACF < 0.05

2. Rotational dynamics (ρ):
   - Delay embedding (dim=6, delay=20 samples)
   - VAR(1) fit with ridge regularization
   - ρ = mean |Im(λ)| / |λ| for eigenvalues λ

FUNCTIONAL TASKS
----------------
1. Temporal Pattern Richness:
   - Brief input pulse, measure trajectory dimensionality
   - Effective dimensionality = 1 / Σ(p_i²) where p_i are normalized eigenvalues
   - High ρ should → richer patterns (higher dimensionality)

2. Integration Window:
   - Brief impulse, measure response decay time
   - Time for response to decay to 50% of peak
   - High τ should → longer integration

RESULTS
-------
With g_I varied from 0.3 to 3.5:

1. τ-ρ Trade-off: r = -0.84, p = 0.0006
   - Strong inhibition: τ ≈ 12ms, ρ ≈ 0.030
   - Weak inhibition: τ ≈ 60ms, ρ ≈ 0.018

2. ρ → Pattern Richness: r = 0.87, p = 0.0003
   - High ρ networks generate richer temporal dynamics

3. τ → Integration Window: r = 0.86, p = 0.0003
   - High τ networks integrate over longer time windows

INTERPRETATION
--------------
The τ-ρ trade-off emerges from E-I balance because:

1. Strong inhibition creates fast negative feedback loops that promote
   oscillatory (rotational) dynamics but prevent sustained activity.

2. Weak inhibition allows recurrent excitation to sustain activity over
   longer periods but reduces the oscillatory component.

This maps onto cortical organization:
- Ventral cortex: PV-rich → strong inhibition → high ρ, low τ → sequences
- Dorsal cortex: SST-rich → weak inhibition → low ρ, high τ → integration

REFERENCES
----------
- Wilson HR, Cowan JD (1972) Excitatory and inhibitory interactions in
  localized populations of model neurons. Biophys J 12:1-24.
- Hennequin G et al. (2014) Optimal control of transient dynamics in
  balanced networks supports generation of complex movements. Neuron 82:1394.
- Murray JD et al. (2014) A hierarchy of intrinsic timescales across
  primate cortex. Nat Neurosci 17:1661-1663.

Author: Generated for Salardini et al. "Dorsoventral Gradient of Rotational Dynamics"
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# NETWORK MODEL
# =============================================================================

class EINetwork:
    """
    Rate-based excitatory-inhibitory recurrent neural network.
    
    Parameters
    ----------
    N_E : int
        Number of excitatory units (default: 100)
    N_I : int
        Number of inhibitory units (default: 25)
    g_I : float
        Inhibitory gain - scales all inhibitory connections (default: 1.0)
        This is the KEY parameter that controls the τ-ρ trade-off.
    """
    
    def __init__(self, N_E=100, N_I=25, g_I=1.0):
        self.N_E = N_E
        self.N_I = N_I
        self.N = N_E + N_I
        self.g_I = g_I
        
        # Build weight matrix
        self.W = self._build_weights()
        
    def _build_weights(self):
        """
        Construct weight matrix with E-I structure.
        
        Connection strengths (before scaling by g_I for inhibitory):
        - E→E: 2.0 (strong recurrence)
        - E→I: 1.5
        - I→E: 1.5 * g_I
        - I→I: 0.8 * g_I
        """
        W = np.zeros((self.N, self.N))
        
        # E→E connections (excitatory, strong for dynamics)
        w_EE = 2.0 / self.N_E
        W[:self.N_E, :self.N_E] = w_EE * np.abs(np.random.randn(self.N_E, self.N_E) * 0.3 + 1)
        
        # E→I connections (excitatory)
        w_IE = 1.5 / self.N_E
        W[self.N_E:, :self.N_E] = w_IE * np.abs(np.random.randn(self.N_I, self.N_E) * 0.3 + 1)
        
        # I→E connections (inhibitory, scaled by g_I)
        w_EI = 1.5 / self.N_I
        W[:self.N_E, self.N_E:] = -self.g_I * w_EI * np.abs(np.random.randn(self.N_E, self.N_I) * 0.3 + 1)
        
        # I→I connections (inhibitory, scaled by g_I)
        w_II = 0.8 / self.N_I
        W[self.N_E:, self.N_E:] = -self.g_I * w_II * np.abs(np.random.randn(self.N_I, self.N_I) * 0.3 + 1)
        
        # No self-connections
        np.fill_diagonal(W, 0)
        
        return W
    
    def _activation(self, x):
        """Softplus activation function (smooth ReLU)."""
        return np.log(1 + np.exp(x)) * 0.5
    
    def simulate(self, T=2000, dt=0.5, input_signal=None):
        """
        Simulate network dynamics.
        
        Parameters
        ----------
        T : float
            Total simulation time in ms
        dt : float
            Time step in ms
        input_signal : ndarray or None
            External input (n_steps x N). If None, constant baseline.
            
        Returns
        -------
        r : ndarray
            Firing rates over time (n_steps x N)
        """
        n_steps = int(T / dt)
        
        # Initialize rates
        r = np.zeros((n_steps, self.N))
        r[0] = np.random.rand(self.N) * 0.2
        
        # Time constants (ms)
        tau_E, tau_I = 10.0, 5.0
        tau = np.concatenate([np.ones(self.N_E) * tau_E, 
                              np.ones(self.N_I) * tau_I])
        
        # Noise level
        noise_std = 0.05
        
        # Simulate
        for i in range(1, n_steps):
            # External input
            if input_signal is not None:
                I_ext = input_signal[min(i, len(input_signal)-1)]
            else:
                I_ext = 1.0  # Constant baseline
            
            # Noise
            noise = noise_std * np.random.randn(self.N) * np.sqrt(dt)
            
            # Rate dynamics: τ * dr/dt = -r + f(Wr + I + noise)
            total_input = self.W @ r[i-1] + I_ext + noise
            dr = (-r[i-1] + self._activation(total_input)) * dt / tau
            
            # Update with clipping
            r[i] = np.clip(r[i-1] + dr, 0, 5)
        
        return r


# =============================================================================
# METRICS
# =============================================================================

def compute_tau(r, dt=0.5):
    """
    Compute intrinsic timescale from population activity.
    
    τ = integral of autocorrelation function until it decays below 0.05.
    Uses only excitatory population (first 80% of units).
    """
    # Population mean (E cells only)
    x = np.mean(r[:, :-25], axis=1)  # Exclude I cells
    x = (x - x.mean()) / (x.std() + 1e-10)
    
    n = len(x)
    max_lag = min(600, n // 3)
    
    # Autocorrelation
    acf = np.correlate(x, x, mode='full')[n-1:n-1+max_lag]
    acf = acf / (acf[0] + 1e-10)
    
    # Truncate at threshold
    zero_crossing = np.where(acf < 0.05)[0]
    if len(zero_crossing) > 0:
        acf = acf[:zero_crossing[0]]
    
    # Integrate
    tau = np.trapz(np.maximum(acf, 0)) * dt
    return max(tau, dt)


def compute_rho(r, dt=0.5, embed_dim=6, embed_delay_ms=10):
    """
    Compute rotational dynamics index via delay embedding + VAR(1).
    
    Parameters
    ----------
    r : ndarray
        Firing rates (n_steps x N)
    embed_dim : int
        Embedding dimension
    embed_delay_ms : float
        Embedding delay in ms
        
    Returns
    -------
    rho : float
        Rotational index (mean |sin(eigenvalue angle)|)
    """
    # Population mean (E cells only)
    x = np.mean(r[:, :-25], axis=1)
    n = len(x)
    
    embed_delay = max(1, int(embed_delay_ms / dt))
    
    if n < embed_dim * embed_delay + 50:
        return np.nan
    
    # Standardize
    x = (x - x.mean()) / (x.std() + 1e-10)
    
    # Delay embedding
    n_embedded = n - (embed_dim - 1) * embed_delay
    X = np.zeros((n_embedded, embed_dim))
    for d in range(embed_dim):
        X[:, d] = x[d * embed_delay : d * embed_delay + n_embedded]
    
    # VAR(1) fit with ridge regularization
    X_past = X[:-1]
    X_future = X[1:]
    ridge_alpha = 0.001
    A = np.linalg.solve(X_past.T @ X_past + ridge_alpha * np.eye(embed_dim), 
                        X_past.T @ X_future)
    
    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues = eigenvalues[np.abs(eigenvalues) > 0.05]  # Threshold small eigenvalues
    
    if len(eigenvalues) == 0:
        return np.nan
    
    # ρ = mean rotational component
    rho = np.mean(np.abs(np.imag(eigenvalues)) / (np.abs(eigenvalues) + 1e-10))
    return rho


# =============================================================================
# FUNCTIONAL TASKS
# =============================================================================

def temporal_pattern_richness(network, n_trials=5):
    """
    Measure richness of temporal patterns via effective dimensionality.
    
    High ρ networks should produce more diverse trajectories.
    
    Returns
    -------
    eff_dim : float
        Effective dimensionality (participation ratio of covariance eigenvalues)
    """
    dims = []
    
    for trial in range(n_trials):
        np.random.seed(trial * 77)
        
        # Create input: brief pulse to random subset
        n_steps = 600
        input_signal = np.ones((n_steps, network.N)) * 0.8
        pulse_neurons = np.random.choice(network.N_E, 20, replace=False)
        input_signal[50:70, pulse_neurons] = 2.0
        
        # Simulate
        r = network.simulate(T=300, input_signal=input_signal)
        
        # Extract E-cell activity after transient
        activity = r[100:, :network.N_E]
        activity = activity - activity.mean(axis=0)
        
        # Covariance and eigenvalues
        cov = np.cov(activity.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = np.maximum(eigs, 0)
        
        # Effective dimensionality (participation ratio)
        eigs_norm = eigs / (eigs.sum() + 1e-10)
        eff_dim = 1 / (np.sum(eigs_norm**2) + 1e-10)
        dims.append(eff_dim)
    
    return np.mean(dims)


def integration_window(network):
    """
    Measure temporal integration via impulse response decay.
    
    High τ networks should maintain responses longer.
    
    Returns
    -------
    duration : float
        Time (ms) for response to decay to 50% of peak
    """
    # Create impulse input
    n_steps = 1000
    input_signal = np.ones((n_steps, network.N)) * 0.8
    input_signal[100:120, :network.N_E//2] = 2.0  # Brief pulse to half of E cells
    
    # Simulate
    r = network.simulate(T=500, input_signal=input_signal)
    
    # Measure response in stimulated population
    response = np.mean(r[:, :network.N_E//2], axis=1)
    baseline = np.mean(response[:100])
    
    # Find peak
    peak_idx = np.argmax(response[100:]) + 100
    peak = response[peak_idx]
    
    if peak <= baseline:
        return 10.0  # Minimum
    
    # Half-life: time to decay to 50% of peak-baseline
    threshold = baseline + (peak - baseline) * 0.5
    decay_indices = np.where(response[peak_idx:] < threshold)[0]
    
    if len(decay_indices) > 0:
        duration = decay_indices[0] * 0.5  # Convert to ms
    else:
        duration = 200.0  # Maximum
    
    return duration


# =============================================================================
# MAIN SIMULATION
# =============================================================================

def run_simulation(g_I_values=None, n_repeats=8, verbose=True):
    """
    Run full parameter sweep.
    
    Parameters
    ----------
    g_I_values : array-like
        Inhibitory gain values to test
    n_repeats : int
        Number of repetitions per g_I value
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Contains g_I, tau, rho, pattern, integ arrays
    """
    if g_I_values is None:
        g_I_values = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0, 3.5])
    
    results = {k: [] for k in ['g_I', 'tau', 'rho', 'pattern', 'integ']}
    
    for g_I in g_I_values:
        if verbose:
            print(f"  g_I = {g_I:.2f}...", end=" ", flush=True)
        
        taus, rhos, patterns, integs = [], [], [], []
        
        for rep in range(n_repeats):
            np.random.seed(rep * 1000 + int(g_I * 100))
            
            # Create network
            net = EINetwork(g_I=g_I)
            
            # Spontaneous activity
            r = net.simulate(T=2000)
            r_steady = r[400:]  # Skip transient
            
            # Compute metrics
            taus.append(compute_tau(r_steady))
            rhos.append(compute_rho(r_steady))
            patterns.append(temporal_pattern_richness(net))
            integs.append(integration_window(net))
        
        # Store means
        results['g_I'].append(g_I)
        results['tau'].append(np.mean(taus))
        results['rho'].append(np.nanmean(rhos))
        results['pattern'].append(np.mean(patterns))
        results['integ'].append(np.mean(integs))
        
        if verbose:
            print(f"τ={np.mean(taus):.1f}ms, ρ={np.nanmean(rhos):.3f}")
    
    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])
    
    return results


def compute_statistics(results):
    """Compute correlations between metrics."""
    tau = results['tau']
    rho = results['rho']
    pattern = results['pattern']
    integ = results['integ']
    
    valid = ~np.isnan(rho)
    
    stats_dict = {}
    stats_dict['tau_rho'] = stats.pearsonr(tau[valid], rho[valid])
    stats_dict['rho_pattern'] = stats.pearsonr(rho[valid], pattern[valid])
    stats_dict['tau_integ'] = stats.pearsonr(tau[valid], integ[valid])
    
    return stats_dict


def create_figure(results, stats_dict, output_path=None):
    """Generate publication figure."""
    
    g_I = results['g_I']
    tau = results['tau']
    rho = results['rho']
    pattern = results['pattern']
    integ = results['integ']
    
    r1, p1 = stats_dict['tau_rho']
    r2, p2 = stats_dict['rho_pattern']
    r3, p3 = stats_dict['tau_integ']
    
    # Figure setup
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9)
    
    # Panel A: Effect of inhibition
    ax = axes[0, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(g_I, tau, 'o-', color='#3498db', lw=2.5, ms=9)
    l2, = ax2.plot(g_I, rho, 's-', color='#e74c3c', lw=2.5, ms=9)
    ax.set_xlabel('Inhibitory Gain (g$_I$)', fontsize=12)
    ax.set_ylabel(r'Timescale $\tau$ (ms)', fontsize=12, color='#3498db')
    ax2.set_ylabel(r'Rotation $\rho$', fontsize=12, color='#e74c3c')
    ax.set_title('A. Inhibition Shapes Dynamics', fontsize=13, fontweight='bold', loc='left')
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    ax2.spines['right'].set_visible(True)
    ax.legend([l1, l2], [r'$\tau$ (timescale)', r'$\rho$ (rotation)'], loc='center right', fontsize=10)
    
    # Panel B: τ-ρ trade-off
    ax = axes[0, 1]
    valid = ~np.isnan(rho)
    sc = ax.scatter(tau[valid], rho[valid], c=g_I[valid], cmap='viridis', s=130, edgecolor='k', lw=1.5)
    m, b = np.polyfit(tau[valid], rho[valid], 1)
    xf = np.linspace(tau.min(), tau.max(), 100)
    ax.plot(xf, m*xf + b, 'k--', lw=2.5)
    ax.set_xlabel(r'Timescale $\tau$ (ms)', fontsize=12)
    ax.set_ylabel(r'Rotation $\rho$', fontsize=12)
    ax.set_title('B. τ-ρ Trade-off Emerges', fontsize=13, fontweight='bold', loc='left')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Inhibitory Gain (g$_I$)', fontsize=10)
    sig = '***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else ''
    ax.text(0.95, 0.95, f'r = {r1:.2f}{sig}\np = {p1:.4f}', transform=ax.transAxes,
            va='top', ha='right', bbox=props, fontsize=11)
    
    # Panel C: ρ → pattern richness
    ax = axes[1, 0]
    ax.scatter(rho[valid], pattern[valid], c='#e74c3c', s=130, edgecolor='k', lw=1.5, alpha=0.85)
    m, b = np.polyfit(rho[valid], pattern[valid], 1)
    xf = np.linspace(rho[valid].min(), rho[valid].max(), 100)
    ax.plot(xf, m*xf + b, 'k--', lw=2.5)
    ax.set_xlabel(r'Rotation $\rho$', fontsize=12)
    ax.set_ylabel('Pattern Richness\n(Effective Dimensionality)', fontsize=12)
    ax.set_title('C. High ρ → Richer Dynamics', fontsize=13, fontweight='bold', loc='left')
    sig = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else ''
    ax.text(0.95, 0.05, f'r = {r2:.2f}{sig}\np = {p2:.4f}', transform=ax.transAxes,
            va='bottom', ha='right', bbox=props, fontsize=11)
    
    # Panel D: τ → integration
    ax = axes[1, 1]
    ax.scatter(tau[valid], integ[valid], c='#3498db', s=130, edgecolor='k', lw=1.5, alpha=0.85)
    m, b = np.polyfit(tau[valid], integ[valid], 1)
    xf = np.linspace(tau.min(), tau.max(), 100)
    ax.plot(xf, m*xf + b, 'k--', lw=2.5)
    ax.set_xlabel(r'Timescale $\tau$ (ms)', fontsize=12)
    ax.set_ylabel('Integration Window (ms)', fontsize=12)
    ax.set_title('D. High τ → Longer Integration', fontsize=13, fontweight='bold', loc='left')
    sig = '***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else ''
    p_str = 'p < 0.001' if p3 < 0.001 else f'p = {p3:.4f}'
    ax.text(0.95, 0.95, f'r = {r3:.2f}{sig}\n{p_str}', transform=ax.transAxes,
            va='top', ha='right', bbox=props, fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path + '.png', dpi=300, facecolor='white', bbox_inches='tight')
        plt.savefig(output_path + '.pdf', facecolor='white', bbox_inches='tight')
        print(f"Saved: {output_path}.png/pdf")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("COMPUTATIONAL MODEL: τ-ρ Trade-off in E-I Networks")
    print("=" * 70)
    
    # Run simulation
    print("\n[1] Running parameter sweep...")
    results = run_simulation(n_repeats=8)
    
    # Statistics
    print("\n[2] Computing statistics...")
    stats_dict = compute_statistics(results)
    
    r1, p1 = stats_dict['tau_rho']
    r2, p2 = stats_dict['rho_pattern']
    r3, p3 = stats_dict['tau_integ']
    
    print(f"    τ-ρ trade-off:        r = {r1:.3f}, p = {p1:.4f}")
    print(f"    ρ → pattern richness: r = {r2:.3f}, p = {p2:.4f}")
    print(f"    τ → integration:      r = {r3:.3f}, p = {p3:.4f}")
    
    # Figure
    print("\n[3] Creating figure...")
    fig = create_figure(results, stats_dict, output_path='fig_computational_model')
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
MODEL: Rate-based E-I network (100E + 25I neurons)
MANIPULATION: Inhibitory gain (g_I) from 0.3 to 3.5

KEY FINDINGS:

1. τ-ρ TRADE-OFF EMERGES NATURALLY
   r = {r1:.2f}, p = {p1:.4f}
   Strong inhibition → low τ, high ρ
   Weak inhibition  → high τ, low ρ

2. FUNCTIONAL CONSEQUENCES
   ρ → Pattern richness:   r = {r2:.2f}, p = {p2:.4f}
   τ → Integration window: r = {r3:.2f}, p = {p3:.4f}

INTERPRETATION:
   The τ-ρ trade-off reflects a fundamental constraint on neural computation.
   Inhibitory tone (potentially set by PV/SST interneuron ratios) controls
   the position along this trade-off:
   
   • Ventral cortex (PV-rich, high ρ): Rich temporal dynamics for sequences
   • Dorsal cortex (SST-rich, high τ): Long integration for memory/decisions
""")
    print("=" * 70)
    
    plt.show()
