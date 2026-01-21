#!/usr/bin/env python3
"""
MOUS band-specific rho vs DV analysis - optimized version.
Uses subset of subjects and vectorized operations for speed.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent
MOUS_DIR = DATA_DIR / "MEG_MOUS" / "intermediates"
CENTROIDS_FILE = DATA_DIR.parent / "code" / "atlas" / "schaefer400_centroids.csv"

# Frequency bands
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta_low': (13.0, 20.0),
    'beta_high': (20.0, 30.0),
    'gamma_low': (30.0, 40.0),
    'broadband': None,  # No additional filtering
}

# Parameters
EMBED_DIM = 10
EMBED_DELAY = 1
RIDGE_ALPHA = 0.001
MAG_MIN = 0.01
N_SUBJECTS = 203  # All subjects
N_PERM = 5000


def compute_rho_fast(ts):
    """Compute rho - optimized."""
    m, d = EMBED_DIM, EMBED_DELAY
    T = len(ts)
    T_eff = T - (m - 1) * d

    if T_eff <= 30:
        return np.nan

    # Delay embed
    E = np.zeros((T_eff, m))
    for k in range(m):
        start = (m - 1 - k) * d
        E[:, k] = ts[start:start + T_eff]

    # Standardize
    E = (E - E.mean(0)) / (E.std(0) + 1e-8)

    # VAR(1) with ridge
    X0, X1 = E[:-1], E[1:]
    XtX = X0.T @ X0
    A = np.linalg.solve(XtX + RIDGE_ALPHA * np.eye(m), X0.T @ X1).T

    # Eigenvalues
    lam = np.linalg.eigvals(A)
    mag = np.abs(lam)
    keep = mag > MAG_MIN

    if not np.any(keep):
        return np.nan

    return float(np.mean(np.abs(np.imag(lam[keep])) / mag[keep]))


def process_subject(args):
    """Process one subject for all bands."""
    subj, subj_dir = args
    results = []

    ts_file = subj_dir / "parcel_ts.npy"
    meta_file = subj_dir / "meta.json"

    if not ts_file.exists():
        return results

    ts = np.load(ts_file)
    with open(meta_file) as f:
        meta = json.load(f)

    fs = meta['stage']['sfreq']
    n_parcels = min(ts.shape[0], 400)

    for band_name, freqs in BANDS.items():
        # Filter if needed
        if freqs is not None:
            low, high = freqs
            nyq = fs / 2
            if high > nyq:
                high = nyq * 0.99
            sos = signal.butter(4, [low/nyq, high/nyq], btype='band', output='sos')
            ts_filt = signal.sosfiltfilt(sos, ts[:n_parcels], axis=-1)
        else:
            ts_filt = ts[:n_parcels]

        # Compute rho for each parcel
        for p_idx in range(n_parcels):
            rho = compute_rho_fast(ts_filt[p_idx])
            results.append({
                'subject': subj,
                'band': band_name,
                'parcel_idx': p_idx,
                'rho': rho
            })

    return results


def main():
    print("=" * 70)
    print("MOUS Band-Specific Rho vs DV Analysis (Fast)")
    print("=" * 70)

    # Load centroids
    centroids = pd.read_csv(CENTROIDS_FILE, comment='#')
    print(f"Loaded {len(centroids)} centroids")

    # Get all subjects
    subjects = sorted([d for d in MOUS_DIR.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    print(f"Processing {len(subjects)} subjects...")

    # Process in parallel
    all_results = []
    args_list = [(s.name, s) for s in subjects]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_subject, args): args[0] for args in args_list}
        done = 0
        for future in as_completed(futures):
            done += 1
            print(f"\r  Progress: {done}/{len(subjects)}", end="", flush=True)
            all_results.extend(future.result())

    print(f"\n  Computed {len(all_results)} rho values")

    df = pd.DataFrame(all_results)

    # Correlations by band
    print(f"\n{'Band':<12} {'r':>8} {'p_param':>12} {'p_spin':>10} {'n':>6}")
    print("-" * 52)

    results = []
    for band_name in BANDS.keys():
        df_band = df[df['band'] == band_name]
        parcel_means = df_band.groupby('parcel_idx')['rho'].mean().reset_index()
        parcel_means.columns = ['parcel_idx', 'rho_mean']

        merged = parcel_means.merge(centroids, on='parcel_idx')
        valid = merged[merged['rho_mean'].notna()]

        if len(valid) < 10:
            continue

        rho_vals = valid['rho_mean'].values
        z_vals = valid['z'].values
        r, p_param = stats.pearsonr(rho_vals, z_vals)

        # Spin test
        lh_idx = np.where(valid['hemi'] == 'lh')[0]
        rh_idx = np.where(valid['hemi'] == 'rh')[0]
        null_r = np.zeros(N_PERM)
        np.random.seed(42)
        for i in range(N_PERM):
            perm = np.zeros(len(valid), dtype=int)
            perm[lh_idx] = np.random.permutation(lh_idx)
            perm[rh_idx] = np.random.permutation(rh_idx)
            null_r[i] = stats.pearsonr(rho_vals[perm], z_vals)[0]
        p_spin = np.mean(np.abs(null_r) >= np.abs(r))

        sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
        print(f"{band_name:<12} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {len(valid):>6} {sig}")

        results.append({'band': band_name, 'r': r, 'p_param': p_param, 'p_spin': p_spin, 'n': len(valid)})

    # Summary
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 70)
    strongest = df_res.loc[df_res['r'].abs().idxmax()]
    print(f"Strongest: {strongest['band']} (r = {strongest['r']:.4f}, p_spin = {strongest['p_spin']:.4f})")

    # Save
    df_res.to_csv(DATA_DIR / "mous_band_rho_correlations.csv", index=False)


if __name__ == '__main__':
    main()
