import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from antropy import spectral_entropy, sample_entropy, perm_entropy
import warnings
warnings.filterwarnings('ignore')

# ============================================
# Configuration
# ============================================
axes_path = Path(r'G:\My Drive\HCP_MEG\derivatives_auditory\derivatives_auditory\axes')
save_dir = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline')
n_subjects = 15  # Process 15 subjects for validation
sf = 250  # Sampling frequency (Hz)

# Get subjects
subjects = sorted([d.name for d in axes_path.iterdir() if d.is_dir()])[:n_subjects]
print(f"Processing {len(subjects)} subjects...")

# ============================================
# Process each subject
# ============================================
all_results = []
subject_correlations = []

for i, sub in enumerate(subjects):
    print(f"  [{i+1}/{len(subjects)}] {sub}...", end=" ")

    sub_path = axes_path / sub
    ts_file = sub_path / 'parcel_ts.npy'
    metrics_file = sub_path / 'parcel_metrics.csv'

    if not ts_file.exists() or not metrics_file.exists():
        print("SKIP (missing files)")
        continue

    # Load data
    parcel_ts = np.load(ts_file)  # Shape: (n_parcels, T)
    metrics = pd.read_csv(metrics_file)

    n_parcels, T = parcel_ts.shape
    print(f"({n_parcels} parcels, {T} timepoints)")

    # Compute nonlinear measures for each parcel
    se_vals = []  # Spectral entropy
    sampen_vals = []  # Sample entropy
    pe_vals = []  # Permutation entropy

    for p in range(n_parcels):
        ts = parcel_ts[p, :]

        # Skip if constant or NaN
        if np.std(ts) < 1e-10 or np.any(np.isnan(ts)):
            se_vals.append(np.nan)
            sampen_vals.append(np.nan)
            pe_vals.append(np.nan)
            continue

        # Normalize
        ts_norm = (ts - np.mean(ts)) / np.std(ts)

        # Spectral entropy
        try:
            se = spectral_entropy(ts_norm, sf=sf, method='welch', normalize=True)
        except:
            se = np.nan
        se_vals.append(se)

        # Sample entropy (use subset for speed)
        try:
            # Downsample for speed (sample entropy is O(n^2))
            ts_down = ts_norm[::4]  # Every 4th sample
            sampen = sample_entropy(ts_down, order=2, metric='chebyshev')
        except:
            sampen = np.nan
        sampen_vals.append(sampen)

        # Permutation entropy
        try:
            pe = perm_entropy(ts_norm, order=3, normalize=True)
        except:
            pe = np.nan
        pe_vals.append(pe)

    # Add to metrics
    metrics['spectral_entropy'] = se_vals
    metrics['sample_entropy'] = sampen_vals
    metrics['perm_entropy'] = pe_vals
    metrics['subject'] = sub

    # Store for aggregate
    all_results.append(metrics)

    # Compute within-subject correlations with rho
    valid_mask = ~(np.isnan(metrics['rho']) | np.isnan(metrics['spectral_entropy']))

    if valid_mask.sum() > 10:
        rho_vals = metrics.loc[valid_mask, 'rho'].values
        z_vals = metrics.loc[valid_mask, 'z'].values
        se_sub = metrics.loc[valid_mask, 'spectral_entropy'].values
        sampen_sub = metrics.loc[valid_mask, 'sample_entropy'].values
        pe_sub = metrics.loc[valid_mask, 'perm_entropy'].values

        # Remove any remaining NaNs for each correlation
        def safe_corr(x, y):
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if mask.sum() > 10:
                return pearsonr(x[mask], y[mask])[0]
            return np.nan

        subject_correlations.append({
            'subject': sub,
            'r_rho_se': safe_corr(rho_vals, se_sub),
            'r_rho_sampen': safe_corr(rho_vals, sampen_sub),
            'r_rho_pe': safe_corr(rho_vals, pe_sub),
            'r_z_se': safe_corr(z_vals, se_sub),
            'r_z_sampen': safe_corr(z_vals, sampen_sub),
            'r_z_pe': safe_corr(z_vals, pe_sub),
            'r_z_rho': safe_corr(z_vals, rho_vals),
        })

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# ============================================
# Combine and save parcel-level data
# ============================================
all_data = pd.concat(all_results, ignore_index=True)
all_data.to_csv(save_dir / 'nonlinear_validation.csv', index=False)
print(f"\nSaved parcel-level data: {save_dir / 'nonlinear_validation.csv'}")
print(f"  Shape: {all_data.shape}")

# ============================================
# Group-level summary
# ============================================
corr_df = pd.DataFrame(subject_correlations)

print("\n" + "="*60)
print("GROUP-LEVEL CORRELATIONS (mean +/- SD across subjects)")
print("="*60)

print("\n### Correlations between rho and nonlinear measures ###")
for col in ['r_rho_se', 'r_rho_sampen', 'r_rho_pe']:
    vals = corr_df[col].dropna()
    measure_name = col.replace('r_rho_', '').upper()
    print(f"  rho vs {measure_name:12s}: r = {vals.mean():+.3f} +/- {vals.std():.3f} (n={len(vals)})")

print("\n### Correlations with DV gradient (z-coordinate) ###")
for col in ['r_z_rho', 'r_z_se', 'r_z_sampen', 'r_z_pe']:
    vals = corr_df[col].dropna()
    measure_name = col.replace('r_z_', '').upper()
    print(f"  z vs {measure_name:12s}: r = {vals.mean():+.3f} +/- {vals.std():.3f} (n={len(vals)})")

# Save correlation summary
corr_df.to_csv(save_dir / 'nonlinear_validation_correlations.csv', index=False)
print(f"\nSaved correlation summary: {save_dir / 'nonlinear_validation_correlations.csv'}")

# ============================================
# Create summary table
# ============================================
summary_rows = []
for col, full_name, interpretation in [
    ('r_rho_se', 'Spectral Entropy', 'Frequency-domain complexity'),
    ('r_rho_sampen', 'Sample Entropy', 'Time-domain regularity'),
    ('r_rho_pe', 'Permutation Entropy', 'Ordinal pattern complexity'),
]:
    vals = corr_df[col].dropna()
    summary_rows.append({
        'Measure': full_name,
        'r_with_rho_mean': vals.mean(),
        'r_with_rho_sd': vals.std(),
        'Interpretation': interpretation,
    })

# Add DV gradient correlations
for col, measure in [('r_z_rho', 'rho (VAR rotational)'),
                      ('r_z_se', 'Spectral Entropy'),
                      ('r_z_sampen', 'Sample Entropy'),
                      ('r_z_pe', 'Permutation Entropy')]:
    vals = corr_df[col].dropna()
    summary_rows.append({
        'Measure': f'{measure} vs z',
        'r_with_rho_mean': vals.mean(),
        'r_with_rho_sd': vals.std(),
        'Interpretation': 'DV gradient',
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(save_dir / 'nonlinear_validation_summary.csv', index=False)
print(f"Saved summary table: {save_dir / 'nonlinear_validation_summary.csv'}")

print("\n" + "="*60)
print("INTERPRETATION GUIDE")
print("="*60)
print("""
If rho correlates POSITIVELY with entropy measures:
  -> Higher rho (rotation) = more complex/irregular dynamics

If rho correlates NEGATIVELY with entropy measures:
  -> Higher rho (rotation) = more regular/predictable dynamics
  -> May indicate rho captures oscillatory (low entropy) structure

If nonlinear measures show SAME DV gradient as rho:
  -> Validates rho captures real dynamical structure
  -> Nonlinear methods agree with VAR(1) approach

If nonlinear measures show DIFFERENT DV gradient:
  -> rho captures something unique to linear dynamics
  -> VAR(1) may miss nonlinear structure (or vice versa)
""")
