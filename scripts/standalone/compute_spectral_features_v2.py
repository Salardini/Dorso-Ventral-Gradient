#!/usr/bin/env python3
"""
COMPUTE SPECTRAL FEATURES FROM PARCEL TIME SERIES

Updated with correct path: data/MEG_MOUS/intermediates/sub-*/parcel_ts.npy

This script computes the spectral confounds needed for reviewer response:
1. Spectral exponent (1/f slope) - THE KEY CONFOUND
2. Total power
3. Bandpower ratios (gamma/delta, fast/slow)
4. Peak frequency

COPY-PASTE THIS ENTIRE SCRIPT INTO PYTHON.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
from datetime import datetime

# ============================================
# CONFIGURATION - UPDATED PATH
# ============================================
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
PARCEL_TS_DIR = DATA_DIR / 'MEG_MOUS' / 'intermediates'
OUTPUT_DIR = DATA_DIR / 'spectral_confound_controls'
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters
FS = 200  # Sampling rate
N_PARCELS = 400

# ============================================
# SPECTRAL FUNCTIONS
# ============================================

def compute_psd(ts, fs=FS):
    """Compute power spectral density using Welch's method."""
    nperseg = min(len(ts) // 4, fs * 2)
    if nperseg < 64:
        nperseg = min(len(ts), 256)
    freqs, psd = signal.welch(ts, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd


def compute_spectral_exponent(freqs, psd, freq_range=(2, 40)):
    """
    Compute spectral exponent (1/f slope).
    P(f) ~ 1/f^beta  =>  log(P) = -beta * log(f) + c
    Returns beta (positive = steeper falloff)
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f, p = freqs[mask], psd[mask]
    
    valid = (f > 0) & (p > 0)
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    log_f = np.log10(f[valid])
    log_p = np.log10(p[valid])
    
    slope, intercept, r, p_val, se = stats.linregress(log_f, log_p)
    
    return -slope, r**2  # beta is negative of slope


def compute_bandpower(freqs, psd, band):
    """Compute power in a frequency band."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    return np.trapz(psd[mask], freqs[mask])


def compute_all_spectral_features(ts, fs=FS):
    """Compute all spectral features for one time series."""
    if len(ts) < 100:
        return None
        
    freqs, psd = compute_psd(ts, fs)
    
    features = {}
    
    # Total power (1-40 Hz)
    features['total_power'] = compute_bandpower(freqs, psd, (1, 40))
    
    # Spectral exponent
    features['spectral_exponent'], features['spectral_exponent_r2'] = \
        compute_spectral_exponent(freqs, psd)
    
    # Bandpowers
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40),
    }
    
    for name, band in bands.items():
        features[f'{name}_power'] = compute_bandpower(freqs, psd, band)
    
    # Ratios
    if features['delta_power'] and features['delta_power'] > 0:
        features['gamma_delta_ratio'] = features['gamma_power'] / features['delta_power']
    else:
        features['gamma_delta_ratio'] = np.nan
    
    slow = (features.get('delta_power', 0) or 0) + (features.get('theta_power', 0) or 0)
    fast = (features.get('beta_power', 0) or 0) + (features.get('gamma_power', 0) or 0)
    if slow > 0:
        features['fast_slow_ratio'] = fast / slow
    else:
        features['fast_slow_ratio'] = np.nan
    
    # Peak frequency
    mask = (freqs >= 1) & (freqs <= 40)
    if np.any(mask) and np.any(psd[mask] > 0):
        features['peak_freq'] = freqs[mask][np.argmax(psd[mask])]
    else:
        features['peak_freq'] = np.nan
    
    return features


# ============================================
# MAIN
# ============================================

print("="*60)
print("COMPUTING SPECTRAL FEATURES")
print("="*60)
print(f"Started: {datetime.now()}")

# Find parcel time series files
ts_files = sorted(PARCEL_TS_DIR.glob('sub-*/parcel_ts.npy'))
print(f"\nSearching in: {PARCEL_TS_DIR}")
print(f"Found {len(ts_files)} subjects")

if len(ts_files) == 0:
    print("ERROR: No parcel_ts.npy files found!")
    print("\nChecking directory structure...")
    if PARCEL_TS_DIR.exists():
        subdirs = list(PARCEL_TS_DIR.iterdir())[:5]
        print(f"Subdirectories: {[s.name for s in subdirs]}")
    exit(1)

# Show first few
print(f"First 3 files:")
for f in ts_files[:3]:
    print(f"  {f}")

# Initialize storage
all_features = {p: [] for p in range(N_PARCELS)}
processed = 0
errors = 0

# Process each subject
print(f"\nProcessing {len(ts_files)} subjects...")
for i, ts_file in enumerate(ts_files):
    if i % 25 == 0:
        print(f"  {i+1}/{len(ts_files)}: {ts_file.parent.name}")
    
    try:
        ts_data = np.load(ts_file)  # Shape: (n_parcels, n_timepoints)
        
        if ts_data.ndim == 1:
            # Single time series - skip
            continue
        
        n_parcels_file = min(N_PARCELS, ts_data.shape[0])
        
        for p in range(n_parcels_file):
            features = compute_all_spectral_features(ts_data[p])
            if features is not None:
                all_features[p].append(features)
        
        processed += 1
        
    except Exception as e:
        errors += 1
        if errors <= 3:
            print(f"  Error in {ts_file.parent.name}: {e}")
        continue

print(f"\nProcessed: {processed} subjects")
print(f"Errors: {errors}")

# Average across subjects
print("\nAveraging across subjects...")

results = []
for p in range(N_PARCELS):
    if len(all_features[p]) == 0:
        results.append({'parcel_idx': p, 'n_subjects': 0})
        continue
    
    row = {'parcel_idx': p, 'n_subjects': len(all_features[p])}
    
    # Average each feature
    feature_names = all_features[p][0].keys()
    for feat in feature_names:
        values = [f[feat] for f in all_features[p] if f.get(feat) is not None and not np.isnan(f.get(feat, np.nan))]
        row[feat] = np.mean(values) if values else np.nan
    
    results.append(row)

df = pd.DataFrame(results)

# Save
output_file = OUTPUT_DIR / 'parcel_spectral_features.csv'
df.to_csv(output_file, index=False)
print(f"\nSaved: {output_file}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Parcels: {len(df)}")
print(f"Subjects per parcel (mean): {df['n_subjects'].mean():.1f}")
print(f"Subjects per parcel (min): {df['n_subjects'].min()}")

print(f"\nFeature statistics:")
for col in ['spectral_exponent', 'total_power', 'gamma_delta_ratio', 'fast_slow_ratio']:
    if col in df.columns:
        valid = df[col].dropna()
        print(f"  {col}:")
        print(f"    Range: {valid.min():.4f} to {valid.max():.4f}")
        print(f"    Mean:  {valid.mean():.4f}")

# Quick correlation check
print("\n" + "="*60)
print("QUICK CORRELATION CHECK")
print("="*60)

# Load parcel coordinates
parcel_file = DATA_DIR / 'group' / 'parcel_group_maps.csv'
if parcel_file.exists():
    parcel_df = pd.read_csv(parcel_file)
    
    # Merge
    merged = df.merge(parcel_df[['parcel_idx', 'z', 'rho_mean']], on='parcel_idx')
    
    print("\nCorrelations with DV coordinate (z):")
    for col in ['spectral_exponent', 'total_power', 'gamma_delta_ratio', 'fast_slow_ratio', 'rho_mean']:
        if col in merged.columns:
            valid = merged[[col, 'z']].dropna()
            if len(valid) > 50:
                r, p = stats.pearsonr(valid[col], valid['z'])
                print(f"  {col} vs z: r = {r:.4f}, p = {p:.4f}")
    
    print("\nCorrelations with rho:")
    for col in ['spectral_exponent', 'total_power', 'gamma_delta_ratio', 'fast_slow_ratio']:
        if col in merged.columns:
            valid = merged[[col, 'rho_mean']].dropna()
            if len(valid) > 50:
                r, p = stats.pearsonr(valid[col], valid['rho_mean'])
                print(f"  {col} vs rho: r = {r:.4f}, p = {p:.4f}")

print(f"\n{'='*60}")
print("NEXT STEP: Run spectral_confound_simple.py")
print("Or send me these results and I'll analyze them")
print(f"{'='*60}")
