"""
Stable τ-ρ Trade-off Model for Nature Neuroscience
Fixed numerical stability issues
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

np.random.seed(42)

class Network:
    def __init__(self, gIE=0.5, tau=20):
        self.N, self.Ne, self.tau = 200, 160, tau
        self.W = self._make_W(gIE)
    
    def _make_W(self, gIE):
        W = np.zeros((self.N, self.N))
        # E→E (weak)
        W[:self.Ne, :self.Ne] = np.random.randn(self.Ne, self.Ne) * 0.03 + 0.08
        # I→E (strong, negative)
        W[:self.Ne, self.Ne:] = -np.abs(np.random.randn(self.Ne, self.N-self.Ne) * 0.2 + gIE)
        # E→I
        W[self.Ne:, :self.Ne] = np.random.randn(self.N-self.Ne, self.Ne) * 0.05 + 0.25
        # I→I
        W[self.Ne:, self.Ne:] = -np.abs(np.random.randn(self.N-self.Ne, self.N-self.Ne) * 0.1 + 0.15)
        return W
    
    def run(self, T=2000):
        r = np.zeros((T, self.N))
        r[0] = np.random.randn(self.N) * 0.05
        for t in range(1, T):
            inp = self.W @ r[t-1] + np.random.randn(self.N) * 0.2
            # Soft ReLU to prevent explosion
            phi = np.maximum(0, np.minimum(10, inp))  # Clip at 10
            dr = (-r[t-1] + phi) / self.tau
            r[t] = r[t-1] + dr * 0.5  # Smaller step for stability
            # Hard clip to prevent runaway
            r[t] = np.clip(r[t], -5, 10)
        return r
    
    def get_tau(self, r):
        m = r.mean(1)
        if np.any(np.isnan(m)) or np.any(np.isinf(m)):
            return np.nan
        acf = np.correlate(m-m.mean(), m-m.mean(), 'full')[len(m)-1:][:300]
        if acf[0] <= 0:
            return np.nan
        acf = acf / acf[0]
        return np.trapz(np.maximum(acf, 0))
    
    def get_rho(self, r):
        x = r.mean(1)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            return np.nan
        # Delay embedding
        m = 8  # Reduced dimension
        if len(x) < m * 2:
            return np.nan
        X = np.array([x[i:len(x)-(m-1)+i] for i in range(m)]).T
        if X.shape[0] < 10:
            return np.nan
        # Ridge regression with stronger regularization
        XtX = X[:-1].T @ X[:-1] + 0.01 * np.eye(m)
        A = X[1:].T @ X[:-1] @ np.linalg.inv(XtX)
        # Check for NaN/Inf
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            return np.nan
        ev = eigvals(A)
        ev = ev[np.abs(ev) > 0.01]
        if len(ev) == 0:
            return np.nan
        return np.mean(np.abs(ev.imag) / np.abs(ev))

# Parameter sweep
print("Running stable parameter sweep...")
gIEs = np.linspace(0.15, 0.85, 15)
taus = np.linspace(12, 45, 15)
results = []

for i, g in enumerate(gIEs):
    for j, t in enumerate(taus):
        net = Network(gIE=g, tau=t)
        r = net.run()
        tau_val = net.get_tau(r)
        rho_val = net.get_rho(r)
        if not (np.isnan(tau_val) or np.isnan(rho_val)):
            results.append([tau_val, rho_val, g, t])
    if (i+1) % 3 == 0:
        print(f"  Progress: {i+1}/{len(gIEs)}")

results = np.array(results)
print(f"Valid simulations: {len(results)}/{len(gIEs)*len(taus)}")

tau_vals, rho_vals, gIE_vals, tau_mem = results.T

# Partial correlation
reg = LinearRegression().fit(tau_mem.reshape(-1,1), tau_vals)
tau_r = tau_vals - reg.predict(tau_mem.reshape(-1,1))
reg = LinearRegression().fit(tau_mem.reshape(-1,1), rho_vals)
rho_r = rho_vals - reg.predict(tau_mem.reshape(-1,1))
r, p = pearsonr(tau_r, rho_r)
print(f"\nτ-ρ trade-off: r={r:.3f}, p={p:.4f}")

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trade-off plot
ax = axes[0, 0]
scatter = ax.scatter(tau_vals, rho_vals, c=gIE_vals, cmap='RdYlBu_r', 
                     s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_xlabel('Intrinsic Timescale τ (ms)', fontsize=12)
ax.set_ylabel('Rotational Index ρ', fontsize=12)
ax.set_title(f'A. τ-ρ Trade-off (r={r:.3f}*)', fontsize=13, fontweight='bold')
ax.text(0.05, 0.95, '*controlling for τ_mem', fontsize=9, transform=ax.transAxes,
        verticalalignment='top')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.colorbar(scatter, ax=ax, label='Inhibitory strength (g_IE)')

# Inhibition effect
ax = axes[0, 1]
g_unique = np.unique(gIE_vals)
rho_by_gIE = [rho_vals[gIE_vals==g].mean() for g in g_unique]
ax.plot(g_unique, rho_by_gIE, 'o-', color='darkgreen', linewidth=2.5, markersize=8)
ax.set_xlabel('Inhibitory Strength (g_IE)', fontsize=12)
ax.set_ylabel('Rotational Index ρ', fontsize=12)
ax.set_title('B. Inhibition Increases ρ', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)

# Ventral vs Dorsal comparison
def make_regime(regime):
    if regime == 'ventral':
        return Network(gIE=0.75, tau=12)
    else:
        return Network(gIE=0.25, tau=40)

print("\nGenerating regime comparison...")
net_v = make_regime('ventral')
net_d = make_regime('dorsal')
r_v = net_v.run(T=3000)
r_d = net_d.run(T=3000)
tau_v, rho_v = net_v.get_tau(r_v), net_v.get_rho(r_v)
tau_d, rho_d = net_d.get_tau(r_d), net_d.get_rho(r_d)

print(f"Ventral: τ={tau_v:.1f}ms, ρ={rho_v:.3f}")
print(f"Dorsal:  τ={tau_d:.1f}ms, ρ={rho_d:.3f}")

# Dynamics plot
ax = axes[1, 0]
t = np.arange(1500)
ax.plot(t, r_v[:1500, 0], 'r-', alpha=0.8, linewidth=1.5, label=f'Ventral (ρ={rho_v:.2f})')
ax.plot(t, r_d[:1500, 0], 'b-', alpha=0.8, linewidth=1.5, label=f'Dorsal (ρ={rho_d:.2f})')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Firing Rate', fontsize=12)
ax.set_title('C. Example Dynamics', fontsize=13, fontweight='bold')
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Summary bar chart
ax = axes[1, 1]
metrics = ['ρ\n(Rotational)', 'τ (ms)\n(Timescale)']
ventral_vals = [rho_v, tau_v/50]
dorsal_vals = [rho_d, tau_d/50]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, ventral_vals, width, label='Ventral (high-ρ)', 
               color='red', alpha=0.7)
bars2 = ax.bar(x + width/2, dorsal_vals, width, label='Dorsal (low-ρ)', 
               color='blue', alpha=0.7)
ax.set_ylabel('Normalized Value', fontsize=12)
ax.set_title('D. Network Properties', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('tau_rho_model_stable.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFigure saved: tau_rho_model_stable.png")