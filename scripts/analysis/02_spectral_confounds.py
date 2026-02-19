#!/usr/bin/env python3
"""
MOUS: Test if rho-DV gradient persists after controlling for spectral confounds.

Computes spectral features (gamma power, alpha power, spectral slope) from
parcel time series and tests whether the rho-DV gradient remains after
residualizing out these spectral confounds.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
from tqdm import tqdm
import json

# Paths
DATA_DIR = Path(__file__).parent
MOUS_DIR = DATA_DIR / "MEG_MOUS" / "intermediates"
PARCEL_FILE = DATA_DIR.parent / "paper_submission" / "A_MOUS" / "parcel_group_maps.csv"
OUTPUT_DIR = DATA_DIR / "group"
OUTPUT_DIR.mkdir(exist_ok=True)

N_PERM = 5000
N_SUBJECTS = 50  # Sample for speed (sufficient for group-level spectral estimates)


def compute_psd(ts, fs, nperseg=None):
    """Compute power spectral density using Welch's method."""
    if nperseg is None:
        nperseg = min(len(ts) // 4, int(fs * 2))  # 2-second windows
    freqs, psd = signal.welch(ts, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd


def compute_spectral_features(ts, fs):
    """
    Compute spectral features for a single time series.

    Returns:
        gamma_rel: relative gamma power (30-40 Hz / 1-40 Hz) - note: data is 1-40 Hz filtered
        alpha_rel: relative alpha power (8-13 Hz / 1-40 Hz)
        beta_rel: relative beta power (13-30 Hz / 1-40 Hz)
        spectral_slope: 1/f exponent (fit to log-log spectrum, 2-30 Hz)
    """
    freqs, psd = compute_psd(ts, fs)

    # Frequency masks
    total_mask = (freqs >= 1) & (freqs <= 40)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    gamma_mask = (freqs >= 30) & (freqs <= 40)
    slope_mask = (freqs >= 2) & (freqs <= 30)  # Avoid edges for slope fit

    # Total power
    total_power = np.trapz(psd[total_mask], freqs[total_mask])

    # Relative band powers
    if total_power > 0:
        alpha_rel = np.trapz(psd[alpha_mask], freqs[alpha_mask]) / total_power
        beta_rel = np.trapz(psd[beta_mask], freqs[beta_mask]) / total_power
        gamma_rel = np.trapz(psd[gamma_mask], freqs[gamma_mask]) / total_power
    else:
        alpha_rel = beta_rel = gamma_rel = np.nan

    # Spectral slope (1/f^beta fit in log-log space)
    f_fit = freqs[slope_mask]
    p_fit = psd[slope_mask]

    # Avoid log(0)
    valid = (f_fit > 0) & (p_fit > 0)
    if np.sum(valid) > 10:
        log_f = np.log10(f_fit[valid])
        log_p = np.log10(p_fit[valid])
        slope, intercept = np.polyfit(log_f, log_p, 1)
        spectral_slope = -slope  # Positive slope means steeper 1/f
    else:
        spectral_slope = np.nan

    return {
        'alpha_rel': alpha_rel,
        'beta_rel': beta_rel,
        'gamma_rel': gamma_rel,
        'spectral_slope': spectral_slope,
        'total_power': total_power,
    }


def residualize(y, X):
    """Residualize y with respect to X."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return y - X_design @ beta


def run_spin_test(x, y, hemi, n_perm=N_PERM):
    """Hemisphere-preserving spin permutation test."""
    r_obs, p_param = stats.pearsonr(x, y)

    lh_idx = np.where(hemi == 'lh')[0]
    rh_idx = np.where(hemi == 'rh')[0]

    null_r = np.zeros(n_perm)
    np.random.seed(42)
    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = np.random.permutation(lh_idx)
        perm[rh_idx] = np.random.permutation(rh_idx)
        null_r[i] = stats.pearsonr(x[perm], y)[0]

    p_spin = np.mean(np.abs(null_r) >= np.abs(r_obs))
    return r_obs, p_param, p_spin


def main():
    print("=" * 70)
    print("MOUS: Spectral Confound Analysis")
    print("=" * 70)

    # Load parcel group data
    print("\n[1/5] Loading parcel group data...")
    df_parcels = pd.read_csv(PARCEL_FILE)
    print(f"    Loaded {len(df_parcels)} parcels")

    # Find subjects
    subjects = sorted([d for d in MOUS_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    np.random.seed(42)
    subjects = list(np.random.choice(subjects, size=min(N_SUBJECTS, len(subjects)), replace=False))
    print(f"    Using {len(subjects)} subjects for spectral estimation")

    # Compute spectral features per parcel
    print("\n[2/5] Computing spectral features...")

    all_features = {p: [] for p in range(400)}

    for subj_dir in tqdm(subjects, desc="Subjects"):
        ts_file = subj_dir / "parcel_ts.npy"
        meta_file = subj_dir / "meta.json"

        if not ts_file.exists():
            continue

        ts = np.load(ts_file)[:400]  # First 400 parcels (Schaefer)
        with open(meta_file) as f:
            meta = json.load(f)
        fs = meta['stage']['sfreq']

        for p_idx in range(min(ts.shape[0], 400)):
            features = compute_spectral_features(ts[p_idx], fs)
            all_features[p_idx].append(features)

    # Aggregate to parcel means
    print("\n[3/5] Aggregating spectral features...")
    spectral_data = []
    for p_idx in range(400):
        if all_features[p_idx]:
            means = {
                'parcel_idx': p_idx,
                'alpha_rel': np.nanmean([f['alpha_rel'] for f in all_features[p_idx]]),
                'beta_rel': np.nanmean([f['beta_rel'] for f in all_features[p_idx]]),
                'gamma_rel': np.nanmean([f['gamma_rel'] for f in all_features[p_idx]]),
                'spectral_slope': np.nanmean([f['spectral_slope'] for f in all_features[p_idx]]),
            }
            spectral_data.append(means)

    df_spectral = pd.DataFrame(spectral_data)

    # Merge with parcel data
    df_merged = df_parcels.merge(df_spectral, on='parcel_idx')
    print(f"    Merged {len(df_merged)} parcels with spectral features")

    # Extract arrays
    rho = df_merged['rho_mean'].values
    tau = df_merged['tau_mean'].values
    z = df_merged['z'].values
    hemi = df_merged['hemi'].values
    gamma_rel = df_merged['gamma_rel'].values
    alpha_rel = df_merged['alpha_rel'].values
    beta_rel = df_merged['beta_rel'].values
    spectral_slope = df_merged['spectral_slope'].values
    ts_rms = df_merged['ts_rms_mean'].values

    # Print spectral correlations with rho
    print("\n[4/5] Testing spectral correlations...")
    print(f"\n{'Variable':<20} {'r with rho':>12} {'r with z':>12}")
    print("-" * 46)

    for name, var in [('gamma_rel', gamma_rel), ('alpha_rel', alpha_rel),
                      ('beta_rel', beta_rel), ('spectral_slope', spectral_slope)]:
        r_rho, _ = stats.pearsonr(var, rho)
        r_z, _ = stats.pearsonr(var, z)
        print(f"{name:<20} {r_rho:>12.4f} {r_z:>12.4f}")

    # Residualization models
    print("\n[5/5] Testing residualized rho vs z correlations...")
    print(f"\n{'Model':<40} {'r':>8} {'p_param':>12} {'p_spin':>10}")
    print("-" * 72)

    results = []

    # Model 0: Raw rho vs z
    r, p_param, p_spin = run_spin_test(rho, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho_raw vs z':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'rho_raw', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Model 1: Residualize rho for gamma_rel
    rho_resid = residualize(rho, gamma_rel)
    r, p_param, p_spin = run_spin_test(rho_resid, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho | gamma_rel vs z':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'rho | gamma_rel', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Model 2: Residualize rho for spectral_slope
    rho_resid = residualize(rho, spectral_slope)
    r, p_param, p_spin = run_spin_test(rho_resid, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho | spectral_slope vs z':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'rho | spectral_slope', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Model 3: Residualize rho for gamma_rel + spectral_slope
    confounds = np.column_stack([gamma_rel, spectral_slope])
    rho_resid = residualize(rho, confounds)
    r, p_param, p_spin = run_spin_test(rho_resid, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho | gamma_rel + slope vs z':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'rho | gamma_rel + slope', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Model 4: Residualize rho for all spectral + ts_rms
    confounds = np.column_stack([gamma_rel, alpha_rel, beta_rel, spectral_slope, ts_rms])
    rho_resid = residualize(rho, confounds)
    r, p_param, p_spin = run_spin_test(rho_resid, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho | all_spectral + ts_rms vs z':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'rho | all_spectral + ts_rms', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Model 5: Also residualize z for same confounds (strictest test)
    z_resid = residualize(z, confounds)
    r, p_param, p_spin = run_spin_test(rho_resid, z_resid, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho | confounds vs z | confounds':<40} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results.append({'model': 'both_residualized', 'r': r, 'p_param': p_param, 'p_spin': p_spin})

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "mous_spectral_confounds.csv", index=False)

    # Save merged data with spectral features
    df_merged.to_csv(OUTPUT_DIR / "mous_parcel_with_spectral.csv", index=False)

    print(f"\nSaved: {OUTPUT_DIR / 'mous_spectral_confounds.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'mous_parcel_with_spectral.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    raw_r = results[0]['r']
    final_r = results[-2]['r']  # all confounds model
    print(f"Raw rho vs z:                   r = {raw_r:.4f}")
    print(f"After all spectral confounds:   r = {final_r:.4f}")
    print(f"Retention: {abs(final_r/raw_r)*100:.1f}% of original effect")
    print("=" * 70)


if __name__ == '__main__':
    main()
