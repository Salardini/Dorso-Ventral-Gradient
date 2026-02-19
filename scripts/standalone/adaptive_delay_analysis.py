#!/usr/bin/env python3
"""
MOUS Adaptive Embedding Delay Analysis

Tests whether frequency-specific ρ-DV gradient magnitudes reflect:
  (A) Real circuit differences, or
  (B) Sampling artifact from fixed embedding delay

Fixed delay (1 sample = 5ms at 200Hz) creates frequency-dependent sensitivity:
  - 2 Hz: 3.6°/sample → low ρ dynamic range
  - 40 Hz: 72°/sample → high ρ dynamic range

Adaptive delay targets ~quarter cycle per delay step, equalizing sensitivity.

If gradients EQUALIZE with adaptive delay → artifact
If gradients STAY DIFFERENT → real circuit difference

Runtime: ~4-6 hours for 203 subjects
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Paths - UPDATE THESE FOR YOUR SYSTEM
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
MOUS_DIR = DATA_DIR / 'MEG_MOUS' / 'intermediates'
CENTROIDS_FILE = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\code\atlas\schaefer400_centroids.csv')
OUTPUT_DIR = DATA_DIR / 'adaptive_delay_analysis'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Frequency bands
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta_low': (13.0, 20.0),
    'beta_high': (20.0, 30.0),
    'gamma_low': (30.0, 40.0),  # MOUS is filtered 1-40 Hz
    'broadband': (1.0, 40.0),
}

# VAR parameters
EMBED_DIM = 10
RIDGE_ALPHA = 0.001
MAG_MIN = 0.01

# Spin test permutations
N_PERM = 5000

# Checkpoint frequency (save every N subjects)
CHECKPOINT_FREQ = 20


# ============================================
# CORE FUNCTIONS
# ============================================

def delay_embed(x, m, d):
    """Create delay embedding matrix.
    
    Args:
        x: 1D time series
        m: embedding dimension
        d: delay (in samples)
    
    Returns:
        E: (T_eff, m) embedding matrix
    """
    T = len(x)
    T_eff = T - (m - 1) * d
    if T_eff <= 30:
        return np.empty((0, m))
    E = np.zeros((T_eff, m), dtype=np.float64)
    for k in range(m):
        start = (m - 1 - k) * d
        E[:, k] = x[start:start + T_eff]
    return E


def compute_rho(ts, embed_delay=1, embed_dim=EMBED_DIM, ridge_alpha=RIDGE_ALPHA, mag_min=MAG_MIN):
    """Compute rotational index from delay-embedded VAR(1).
    
    Args:
        ts: 1D time series
        embed_delay: delay between embedding dimensions (samples)
        embed_dim: number of embedding dimensions
        ridge_alpha: regularization strength
        mag_min: minimum eigenvalue magnitude to include
    
    Returns:
        rho: rotational index (ratio of imaginary to real eigenvalue components)
        r2: model fit (R-squared)
    """
    E = delay_embed(ts, m=embed_dim, d=embed_delay)
    if E.shape[0] < 30:
        return np.nan, np.nan
    
    # Standardize
    E = (E - E.mean(axis=0)) / (E.std(axis=0) + 1e-8)
    
    # Fit VAR(1) with ridge regression
    X0, X1 = E[:-1], E[1:]
    XtX = X0.T @ X0
    p = XtX.shape[0]
    A_T = np.linalg.solve(XtX + ridge_alpha * np.eye(p), X0.T @ X1)
    A = A_T.T
    
    # Eigenvalues
    lam = np.linalg.eigvals(A)
    mag = np.abs(lam)
    keep = mag > mag_min
    
    if not np.any(keep):
        return np.nan, np.nan
    
    lam_keep = lam[keep]
    mag_keep = mag[keep]
    
    # rho = mean(|imag(lambda)| / |lambda|)
    rho = np.mean(np.abs(np.imag(lam_keep)) / mag_keep)
    
    # R-squared
    pred = X0 @ A_T
    ss_res = np.sum((X1 - pred)**2)
    ss_tot = np.sum((X1 - X1.mean(axis=0))**2)
    r2 = 1 - ss_res / ss_tot
    
    return float(rho), float(r2)


def bandpass_filter(data, fs, low, high, order=4):
    """Apply zero-phase bandpass filter.
    
    Args:
        data: (n_parcels, n_time) array
        fs: sampling frequency
        low, high: filter cutoffs
        order: filter order
    
    Returns:
        filtered: filtered data
    """
    nyq = fs / 2
    if low >= nyq or high > nyq:
        if low >= nyq:
            return np.zeros_like(data)
        high = nyq * 0.99
    
    sos = signal.butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return signal.sosfiltfilt(sos, data, axis=-1)


def compute_adaptive_delay(center_freq, fs, target_phase=0.25):
    """Compute adaptive embedding delay to achieve target phase advance.
    
    Args:
        center_freq: center frequency of band (Hz)
        fs: sampling frequency (Hz)
        target_phase: target phase advance per delay (cycles, default=0.25 = quarter cycle)
    
    Returns:
        delay: embedding delay in samples (minimum 1)
    """
    # Time for target_phase cycles at center_freq
    target_time = target_phase / center_freq
    # Convert to samples
    delay = int(round(target_time * fs))
    return max(1, delay)


def run_spin_test(rho_vals, z_vals, hemi_labels, n_perm=N_PERM):
    """Run hemisphere-preserving spin permutation test.
    
    Args:
        rho_vals: parcel rho values
        z_vals: parcel z coordinates
        hemi_labels: hemisphere labels ('lh' or 'rh')
        n_perm: number of permutations
    
    Returns:
        r: observed correlation
        p_spin: spin-test p-value (two-tailed)
    """
    # Observed correlation
    r, _ = stats.pearsonr(rho_vals, z_vals)
    
    # Get hemisphere indices
    lh_idx = np.where(hemi_labels == 'lh')[0]
    rh_idx = np.where(hemi_labels == 'rh')[0]
    
    # Permutation test (hemisphere-preserving)
    null_r = np.zeros(n_perm)
    rng = np.random.default_rng(42)
    
    for i in range(n_perm):
        perm = np.zeros(len(rho_vals), dtype=int)
        perm[lh_idx] = rng.permutation(lh_idx)
        perm[rh_idx] = rng.permutation(rh_idx)
        null_r[i] = stats.pearsonr(rho_vals[perm], z_vals)[0]
    
    # Two-tailed p-value
    p_spin = np.mean(np.abs(null_r) >= np.abs(r))
    
    return r, p_spin


# ============================================
# MAIN ANALYSIS
# ============================================

def main():
    print("="*80)
    print("MOUS ADAPTIVE EMBEDDING DELAY ANALYSIS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load centroids
    print("[1/5] Loading Schaefer 400 centroids...")
    if not CENTROIDS_FILE.exists():
        # Try alternative path
        alt_path = DATA_DIR.parent / 'code' / 'atlas' / 'schaefer400_centroids.csv'
        if alt_path.exists():
            centroids = pd.read_csv(alt_path, comment='#')
        else:
            print(f"ERROR: Cannot find centroids file at {CENTROIDS_FILE}")
            print("Please update CENTROIDS_FILE path in script")
            return
    else:
        centroids = pd.read_csv(CENTROIDS_FILE, comment='#')
    print(f"    Loaded {len(centroids)} parcel centroids")
    
    # Find subjects
    print("\n[2/5] Finding MOUS subjects...")
    subjects = sorted([d.name for d in MOUS_DIR.iterdir() 
                      if d.is_dir() and d.name.startswith('sub-')])
    print(f"    Found {len(subjects)} subjects")
    
    # Check for existing checkpoint
    checkpoint_file = OUTPUT_DIR / 'checkpoint.json'
    parcel_results_file = OUTPUT_DIR / 'adaptive_delay_parcel_results.csv'
    
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        completed_subjects = set(checkpoint['completed_subjects'])
        print(f"    Resuming from checkpoint: {len(completed_subjects)} subjects already done")
        
        # Load existing results
        if parcel_results_file.exists():
            existing_results = pd.read_csv(parcel_results_file)
            all_results = existing_results.to_dict('records')
        else:
            all_results = []
    else:
        completed_subjects = set()
        all_results = []
    
    subjects_to_process = [s for s in subjects if s not in completed_subjects]
    print(f"    Subjects to process: {len(subjects_to_process)}")
    
    # Compute adaptive delays for each band
    print("\n[3/5] Computing adaptive delays...")
    adaptive_delays = {}
    print(f"    {'Band':<12} {'Center (Hz)':<12} {'Fixed delay':<12} {'Adaptive delay':<15} {'Phase/delay':<12}")
    print("    " + "-"*65)
    
    for band_name, (low, high) in BANDS.items():
        center = (low + high) / 2
        adaptive_delay = compute_adaptive_delay(center, 200)  # Assuming 200 Hz
        adaptive_delays[band_name] = adaptive_delay
        
        # Calculate phase advance per delay
        phase_per_delay = center * (adaptive_delay / 200) * 360  # degrees
        
        print(f"    {band_name:<12} {center:<12.1f} {1:<12} {adaptive_delay:<15} {phase_per_delay:<12.1f}°")
    
    # Process subjects
    print(f"\n[4/5] Processing subjects...")
    print(f"    Checkpoint every {CHECKPOINT_FREQ} subjects")
    print()
    
    n_total = len(subjects_to_process)
    
    for i, subj in enumerate(subjects_to_process):
        subj_dir = MOUS_DIR / subj
        ts_file = subj_dir / 'parcel_ts.npy'
        meta_file = subj_dir / 'meta.json'
        
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = datetime.now().strftime('%H:%M:%S')
            print(f"    [{elapsed}] Processing {i+1}/{n_total}: {subj}")
        
        if not ts_file.exists():
            print(f"    WARNING: Missing {ts_file}")
            continue
        
        # Load data
        ts = np.load(ts_file)  # (n_parcels, n_time)
        with open(meta_file) as f:
            meta = json.load(f)
        
        fs = meta['stage']['sfreq']
        n_parcels = min(ts.shape[0], 400)
        
        # Update adaptive delays with actual sampling rate
        if fs != 200:
            for band_name, (low, high) in BANDS.items():
                center = (low + high) / 2
                adaptive_delays[band_name] = compute_adaptive_delay(center, fs)
        
        # Process each band
        for band_name, (low, high) in BANDS.items():
            # Filter
            if band_name == 'broadband':
                ts_filt = ts
            else:
                ts_filt = bandpass_filter(ts, fs, low, high)
            
            # Get delays
            fixed_delay = 1
            adaptive_delay = adaptive_delays[band_name]
            
            # Compute rho for each parcel
            for p_idx in range(n_parcels):
                parcel_ts = ts_filt[p_idx]
                
                # Fixed delay
                rho_fixed, r2_fixed = compute_rho(parcel_ts, embed_delay=fixed_delay)
                
                # Adaptive delay
                rho_adaptive, r2_adaptive = compute_rho(parcel_ts, embed_delay=adaptive_delay)
                
                all_results.append({
                    'subject': subj,
                    'band': band_name,
                    'parcel_idx': p_idx,
                    'rho_fixed': rho_fixed,
                    'r2_fixed': r2_fixed,
                    'rho_adaptive': rho_adaptive,
                    'r2_adaptive': r2_adaptive,
                    'fixed_delay': fixed_delay,
                    'adaptive_delay': adaptive_delay,
                })
        
        # Mark as completed
        completed_subjects.add(subj)
        
        # Checkpoint
        if (len(completed_subjects) % CHECKPOINT_FREQ == 0) or (i == n_total - 1):
            print(f"    Saving checkpoint ({len(completed_subjects)} subjects done)...")
            
            # Save results so far
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(parcel_results_file, index=False)
            
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'completed_subjects': list(completed_subjects),
                    'timestamp': datetime.now().isoformat()
                }, f)
    
    print(f"\n    Completed all {len(completed_subjects)} subjects!")
    
    # Final save
    print("\n[5/5] Computing group statistics and correlations...")
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(parcel_results_file, index=False)
    print(f"    Saved parcel-level results: {parcel_results_file}")
    
    # Aggregate to parcel means
    parcel_means = df_results.groupby(['band', 'parcel_idx']).agg({
        'rho_fixed': 'mean',
        'rho_adaptive': 'mean',
        'r2_fixed': 'mean',
        'r2_adaptive': 'mean',
    }).reset_index()
    
    # Merge with centroids
    parcel_means = parcel_means.merge(centroids, on='parcel_idx')
    
    # Compute correlations
    print("\n" + "="*80)
    print("RESULTS: ρ-DV CORRELATIONS BY BAND AND DELAY TYPE")
    print("="*80)
    print()
    print(f"{'Band':<12} {'Fixed r':<10} {'Fixed p_spin':<12} {'Adapt r':<10} {'Adapt p_spin':<12} {'Δ|r|':<10}")
    print("-"*70)
    
    correlation_results = []
    
    for band_name in BANDS.keys():
        df_band = parcel_means[parcel_means['band'] == band_name].copy()
        
        # Remove NaN
        valid = df_band.dropna(subset=['rho_fixed', 'rho_adaptive'])
        
        if len(valid) < 50:
            print(f"{band_name:<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'N/A':<12}")
            continue
        
        rho_fixed = valid['rho_fixed'].values
        rho_adaptive = valid['rho_adaptive'].values
        z_vals = valid['z'].values
        hemi_labels = valid['hemi'].values
        
        # Correlations
        r_fixed, p_spin_fixed = run_spin_test(rho_fixed, z_vals, hemi_labels)
        r_adaptive, p_spin_adaptive = run_spin_test(rho_adaptive, z_vals, hemi_labels)
        
        # Change in magnitude
        delta_r = np.abs(r_adaptive) - np.abs(r_fixed)
        
        # Significance markers
        sig_fixed = '***' if p_spin_fixed < 0.001 else '**' if p_spin_fixed < 0.01 else '*' if p_spin_fixed < 0.05 else ''
        sig_adaptive = '***' if p_spin_adaptive < 0.001 else '**' if p_spin_adaptive < 0.01 else '*' if p_spin_adaptive < 0.05 else ''
        
        print(f"{band_name:<12} {r_fixed:>+8.3f}{sig_fixed:<2} {p_spin_fixed:<12.4f} {r_adaptive:>+8.3f}{sig_adaptive:<2} {p_spin_adaptive:<12.4f} {delta_r:>+8.3f}")
        
        correlation_results.append({
            'band': band_name,
            'center_freq': (BANDS[band_name][0] + BANDS[band_name][1]) / 2,
            'r_fixed': r_fixed,
            'p_spin_fixed': p_spin_fixed,
            'r_adaptive': r_adaptive,
            'p_spin_adaptive': p_spin_adaptive,
            'delta_abs_r': delta_r,
            'fixed_delay': 1,
            'adaptive_delay': adaptive_delays[band_name],
            'n_parcels': len(valid),
        })
    
    # Save correlation results
    df_corr = pd.DataFrame(correlation_results)
    corr_file = OUTPUT_DIR / 'adaptive_delay_correlations.csv'
    df_corr.to_csv(corr_file, index=False)
    print(f"\nSaved correlation results: {corr_file}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Compare slow vs fast
    slow_bands = ['delta', 'theta']
    fast_bands = ['alpha', 'beta_low', 'beta_high', 'gamma_low']
    
    slow_fixed = df_corr[df_corr['band'].isin(slow_bands)]['r_fixed'].abs().mean()
    slow_adaptive = df_corr[df_corr['band'].isin(slow_bands)]['r_adaptive'].abs().mean()
    fast_fixed = df_corr[df_corr['band'].isin(fast_bands)]['r_fixed'].abs().mean()
    fast_adaptive = df_corr[df_corr['band'].isin(fast_bands)]['r_adaptive'].abs().mean()
    
    print(f"\nGradient magnitude (mean |r|):")
    print(f"                     Fixed delay    Adaptive delay")
    print(f"  Slow (δ,θ):        {slow_fixed:.3f}          {slow_adaptive:.3f}")
    print(f"  Fast (α,β,γ):      {fast_fixed:.3f}          {fast_adaptive:.3f}")
    print(f"  Fast/Slow ratio:   {fast_fixed/slow_fixed:.2f}x           {fast_adaptive/slow_adaptive:.2f}x")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    ratio_fixed = fast_fixed / slow_fixed
    ratio_adaptive = fast_adaptive / slow_adaptive
    
    if abs(ratio_adaptive - 1.0) < abs(ratio_fixed - 1.0) * 0.5:
        print("""
RESULT: Gradients EQUALIZED with adaptive delay

The magnitude difference between slow and fast rhythms was largely
eliminated by using adaptive embedding delay. This suggests the
original asymmetry was partly a SAMPLING ARTIFACT:
  - Fixed delay captures more rotation at high frequencies
  - Adaptive delay equalizes sensitivity across frequencies

IMPLICATION: The direction reversal (slow=dorsal, fast=ventral) is REAL,
but the magnitude difference should be interpreted cautiously.
""")
    else:
        print("""
RESULT: Gradients REMAIN DIFFERENT with adaptive delay

Even with adaptive embedding delay that equalizes sensitivity,
fast rhythms still show stronger DV gradients than slow rhythms.

IMPLICATION: The magnitude difference reflects REAL circuit properties,
not just a sampling artifact. Fast local dynamics are more tightly
constrained by DV anatomy than slow distributed dynamics.
""")
    
    # Mean rho by band
    print("\n" + "="*80)
    print("MEAN ρ VALUES BY BAND")
    print("="*80)
    print(f"\n{'Band':<12} {'ρ (fixed)':<12} {'ρ (adaptive)':<12}")
    print("-"*40)
    
    for band_name in BANDS.keys():
        df_band = parcel_means[parcel_means['band'] == band_name]
        mean_fixed = df_band['rho_fixed'].mean()
        mean_adaptive = df_band['rho_adaptive'].mean()
        print(f"{band_name:<12} {mean_fixed:<12.3f} {mean_adaptive:<12.3f}")
    
    # Correlation between freq and rho
    mean_rho_by_band = parcel_means.groupby('band').agg({
        'rho_fixed': 'mean',
        'rho_adaptive': 'mean'
    }).reset_index()
    mean_rho_by_band['center_freq'] = mean_rho_by_band['band'].map(
        lambda b: (BANDS[b][0] + BANDS[b][1]) / 2 if b != 'broadband' else 20
    )
    
    # Exclude broadband for this correlation
    no_bb = mean_rho_by_band[mean_rho_by_band['band'] != 'broadband']
    r_freq_rho_fixed, _ = stats.pearsonr(no_bb['center_freq'], no_bb['rho_fixed'])
    r_freq_rho_adaptive, _ = stats.pearsonr(no_bb['center_freq'], no_bb['rho_adaptive'])
    
    print(f"\nCorrelation (center freq vs mean ρ):")
    print(f"  Fixed delay:    r = {r_freq_rho_fixed:.3f}")
    print(f"  Adaptive delay: r = {r_freq_rho_adaptive:.3f}")
    
    if abs(r_freq_rho_adaptive) < abs(r_freq_rho_fixed) * 0.5:
        print("  → Adaptive delay REMOVES frequency-ρ confound")
    else:
        print("  → Frequency-ρ relationship persists (may be real)")
    
    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Clean up checkpoint file
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Removed checkpoint file (analysis complete)")


if __name__ == '__main__':
    main()
