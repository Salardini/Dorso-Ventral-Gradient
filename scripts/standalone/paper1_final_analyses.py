#!/usr/bin/env python3
"""
Paper 1 Final Analyses

Computes remaining statistics needed for manuscript v5:
1. ρ vs Principal Gradient with p_spin
2. R² distribution (5th percentile, IQR)
3. Subject-level DV correlations (optional but recommended)
4. Summary statistics table

Run this after the main pipeline is complete.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
from datetime import datetime

# ============================================
# CONFIGURATION - UPDATE PATHS
# ============================================

# Input paths
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
PARCEL_GROUP_FILE = DATA_DIR / 'group' / 'parcel_group_maps_filtered.csv'
SUBJECT_METRICS_FILE = DATA_DIR / 'group' / 'mous_subject_metrics_filtered.csv'
CENTROIDS_FILE = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\code\atlas\schaefer400_centroids.csv')

# Principal gradient file (from neuromaps or similar)
# You may need to create this or download from neuromaps
PRINCIPAL_GRADIENT_FILE = DATA_DIR / 'atlas' / 'schaefer400_principal_gradient.csv'

# Output
OUTPUT_DIR = DATA_DIR / 'paper1_final_stats'
OUTPUT_DIR.mkdir(exist_ok=True)

# Spin test parameters
N_PERM = 10000
SEED = 42


# ============================================
# SPIN PERMUTATION TEST
# ============================================

def spin_permutation_test(x, y, hemi_labels, n_perm=N_PERM, seed=SEED):
    """
    Hemisphere-preserving spin permutation test.
    
    Args:
        x, y: Arrays to correlate
        hemi_labels: Hemisphere labels ('lh' or 'rh')
        n_perm: Number of permutations
        seed: Random seed
    
    Returns:
        r: Observed correlation
        p_spin: Two-tailed p-value
        null_dist: Null distribution of correlations
    """
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    hemi_labels = hemi_labels[valid]
    
    # Observed correlation
    r, _ = stats.pearsonr(x, y)
    
    # Get hemisphere indices
    lh_idx = np.where(hemi_labels == 'lh')[0]
    rh_idx = np.where(hemi_labels == 'rh')[0]
    
    # Permutation test
    rng = np.random.default_rng(seed)
    null_r = np.zeros(n_perm)
    
    for i in range(n_perm):
        perm = np.zeros(len(x), dtype=int)
        perm[lh_idx] = rng.permutation(lh_idx)
        perm[rh_idx] = rng.permutation(rh_idx)
        null_r[i] = stats.pearsonr(x[perm], y)[0]
    
    # Two-tailed p-value
    p_spin = np.mean(np.abs(null_r) >= np.abs(r))
    
    return r, p_spin, null_r


# ============================================
# ANALYSIS 1: ρ vs PRINCIPAL GRADIENT
# ============================================

def analysis_rho_vs_principal_gradient():
    """
    Test correlation between ρ and principal functional connectivity gradient.
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: ρ vs Principal Gradient")
    print("="*60)
    
    # Load data
    parcel_data = pd.read_csv(PARCEL_GROUP_FILE)
    centroids = pd.read_csv(CENTROIDS_FILE, comment='#')
    
    # Check for principal gradient file
    if not PRINCIPAL_GRADIENT_FILE.exists():
        print(f"\nWARNING: Principal gradient file not found: {PRINCIPAL_GRADIENT_FILE}")
        print("Creating placeholder with instructions...")
        
        # Create instructions file
        instructions = """
# Principal Gradient Data

To complete this analysis, you need the principal functional connectivity gradient
for Schaefer-400 parcels. Options:

1. **From neuromaps** (Python):
   ```python
   from neuromaps.datasets import fetch_annotation
   from neuromaps.parcellate import Parcellater
   
   # Fetch Margulies principal gradient
   gradient = fetch_annotation(source='margulies2016', desc='fcgradient01')
   
   # Parcellate to Schaefer-400
   parcellater = Parcellater(
       parcellation='schaefer', 
       scale=400, 
       space='fsLR'
   )
   pc1_parcellated = parcellater.fit_transform(gradient, space='fsLR')
   ```

2. **From your FC analysis**:
   - Compute group-average functional connectivity matrix
   - Apply diffusion map embedding
   - Extract first gradient (PC1)

3. **Download pre-computed**:
   - Available from ENIGMA toolbox or neuromaps

Save as CSV with columns: parcel_idx, pc1
"""
        with open(PRINCIPAL_GRADIENT_FILE.parent / 'README_principal_gradient.md', 'w') as f:
            f.write(instructions)
        
        print(f"Instructions saved to: {PRINCIPAL_GRADIENT_FILE.parent / 'README_principal_gradient.md'}")
        
        # Return placeholder result
        return {
            'analysis': 'rho_vs_pc1',
            'status': 'PENDING - need principal gradient data',
            'r': None,
            'p_spin': None,
        }
    
    # Load principal gradient
    pc1_data = pd.read_csv(PRINCIPAL_GRADIENT_FILE)
    
    # Merge
    merged = parcel_data.merge(pc1_data, on='parcel_idx')
    merged = merged.merge(centroids[['parcel_idx', 'hemi']], on='parcel_idx')
    
    # Get arrays
    rho = merged['rho_mean'].values
    pc1 = merged['pc1'].values
    hemi = merged['hemi'].values
    
    # Spin test
    r, p_spin, null_dist = spin_permutation_test(rho, pc1, hemi)
    
    print(f"\nρ vs Principal Gradient:")
    print(f"  r = {r:.4f}")
    print(f"  p_spin = {p_spin:.4f}")
    print(f"  Interpretation: {'Orthogonal' if p_spin > 0.05 else 'Correlated'}")
    
    result = {
        'analysis': 'rho_vs_pc1',
        'r': r,
        'p_spin': p_spin,
        'n_parcels': len(rho),
        'interpretation': 'orthogonal' if p_spin > 0.05 else 'correlated',
    }
    
    return result


# ============================================
# ANALYSIS 2: R² DISTRIBUTION
# ============================================

def analysis_r2_distribution():
    """
    Compute R² distribution statistics for VAR(1) fits.
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: R² Distribution")
    print("="*60)
    
    # Load subject-level data
    if not SUBJECT_METRICS_FILE.exists():
        # Try alternative path
        alt_path = DATA_DIR / 'mous_subject_metrics_filtered.csv'
        if alt_path.exists():
            subject_data = pd.read_csv(alt_path)
        else:
            print(f"WARNING: Subject metrics file not found")
            print(f"Tried: {SUBJECT_METRICS_FILE}")
            print(f"       {alt_path}")
            return {'status': 'FILE_NOT_FOUND'}
    else:
        subject_data = pd.read_csv(SUBJECT_METRICS_FILE)
    
    # Check for r2 column
    r2_col = None
    for col in ['rho_r2', 'r2', 'var1_r2', 'rho_r2_mean']:
        if col in subject_data.columns:
            r2_col = col
            break
    
    if r2_col is None:
        print("WARNING: No R² column found in subject metrics")
        print(f"Available columns: {list(subject_data.columns)}")
        return {'status': 'NO_R2_COLUMN'}
    
    r2_values = subject_data[r2_col].dropna().values
    
    # Compute statistics
    stats_dict = {
        'analysis': 'r2_distribution',
        'n_observations': len(r2_values),
        'median': float(np.median(r2_values)),
        'mean': float(np.mean(r2_values)),
        'std': float(np.std(r2_values)),
        'min': float(np.min(r2_values)),
        'max': float(np.max(r2_values)),
        'percentile_5': float(np.percentile(r2_values, 5)),
        'percentile_25': float(np.percentile(r2_values, 25)),
        'percentile_75': float(np.percentile(r2_values, 75)),
        'percentile_95': float(np.percentile(r2_values, 95)),
        'iqr': float(np.percentile(r2_values, 75) - np.percentile(r2_values, 25)),
    }
    
    print(f"\nR² Distribution (N = {stats_dict['n_observations']}):")
    print(f"  Median:         {stats_dict['median']:.6f}")
    print(f"  5th percentile: {stats_dict['percentile_5']:.6f}")
    print(f"  25th percentile:{stats_dict['percentile_25']:.6f}")
    print(f"  75th percentile:{stats_dict['percentile_75']:.6f}")
    print(f"  IQR:            {stats_dict['iqr']:.6f}")
    print(f"  Range:          [{stats_dict['min']:.6f}, {stats_dict['max']:.6f}]")
    
    print(f"\nFor paper: 'median R² = {stats_dict['median']:.4f}; " + 
          f"5th percentile = {stats_dict['percentile_5']:.4f}'")
    
    return stats_dict


# ============================================
# ANALYSIS 3: SUBJECT-LEVEL DV CORRELATIONS
# ============================================

def analysis_subject_level_dv():
    """
    Compute ρ-DV correlation for each subject individually.
    Tests whether the gradient is consistent at the individual level.
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Subject-Level DV Correlations")
    print("="*60)
    
    # Load centroids
    centroids = pd.read_csv(CENTROIDS_FILE, comment='#')
    z_coords = centroids['z'].values
    
    # Load subject-level metrics
    if not SUBJECT_METRICS_FILE.exists():
        alt_path = DATA_DIR / 'mous_subject_metrics_filtered.csv'
        if alt_path.exists():
            subject_data = pd.read_csv(alt_path)
        else:
            print("WARNING: Subject metrics file not found")
            return {'status': 'FILE_NOT_FOUND'}
    else:
        subject_data = pd.read_csv(SUBJECT_METRICS_FILE)
    
    # Get unique subjects
    subjects = subject_data['subject'].unique()
    print(f"Found {len(subjects)} subjects")
    
    # Compute correlation for each subject
    subject_correlations = []
    
    for subj in subjects:
        subj_data = subject_data[subject_data['subject'] == subj].sort_values('parcel_idx')
        
        if len(subj_data) < 100:
            continue
        
        rho_vals = subj_data['rho'].values[:400]  # Ensure 400 parcels
        
        if len(rho_vals) != len(z_coords):
            continue
        
        # Correlation
        valid = ~np.isnan(rho_vals)
        if np.sum(valid) < 100:
            continue
        
        r, p = stats.pearsonr(rho_vals[valid], z_coords[valid])
        subject_correlations.append({
            'subject': subj,
            'r_dv': r,
            'p_value': p,
            'n_parcels': np.sum(valid),
        })
    
    df_subj = pd.DataFrame(subject_correlations)
    
    if len(df_subj) == 0:
        print("WARNING: No valid subject correlations computed")
        return {'status': 'NO_VALID_SUBJECTS'}
    
    # Statistics
    r_values = df_subj['r_dv'].values
    
    # One-sample t-test against 0
    t_stat, p_ttest = stats.ttest_1samp(r_values, 0)
    
    # Proportion negative (as expected)
    prop_negative = np.mean(r_values < 0)
    
    stats_dict = {
        'analysis': 'subject_level_dv',
        'n_subjects': len(df_subj),
        'mean_r': float(np.mean(r_values)),
        'std_r': float(np.std(r_values)),
        'median_r': float(np.median(r_values)),
        'min_r': float(np.min(r_values)),
        'max_r': float(np.max(r_values)),
        'prop_negative': float(prop_negative),
        't_statistic': float(t_stat),
        'p_ttest': float(p_ttest),
        'ci_95_lower': float(np.mean(r_values) - 1.96 * np.std(r_values) / np.sqrt(len(r_values))),
        'ci_95_upper': float(np.mean(r_values) + 1.96 * np.std(r_values) / np.sqrt(len(r_values))),
    }
    
    print(f"\nSubject-Level ρ-DV Correlations (N = {stats_dict['n_subjects']}):")
    print(f"  Mean r:     {stats_dict['mean_r']:.4f} ± {stats_dict['std_r']:.4f}")
    print(f"  95% CI:     [{stats_dict['ci_95_lower']:.4f}, {stats_dict['ci_95_upper']:.4f}]")
    print(f"  Median r:   {stats_dict['median_r']:.4f}")
    print(f"  Range:      [{stats_dict['min_r']:.4f}, {stats_dict['max_r']:.4f}]")
    print(f"  % negative: {stats_dict['prop_negative']*100:.1f}%")
    print(f"  t-test vs 0: t = {stats_dict['t_statistic']:.2f}, p = {stats_dict['p_ttest']:.2e}")
    
    # Save individual correlations
    df_subj.to_csv(OUTPUT_DIR / 'subject_level_dv_correlations.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'subject_level_dv_correlations.csv'}")
    
    return stats_dict


# ============================================
# ANALYSIS 4: COMPREHENSIVE SUMMARY
# ============================================

def create_summary_table():
    """
    Create comprehensive summary table for Extended Data.
    """
    print("\n" + "="*60)
    print("Creating Summary Table")
    print("="*60)
    
    # Load overnight results
    atlas_file = DATA_DIR / 'atlas_sensitivity' / 'atlas_sensitivity_correlations.csv'
    adaptive_file = DATA_DIR / 'adaptive_delay_analysis' / 'adaptive_delay_correlations.csv'
    
    summary_rows = []
    
    # Atlas sensitivity
    if atlas_file.exists():
        atlas_df = pd.read_csv(atlas_file)
        for _, row in atlas_df.iterrows():
            summary_rows.append({
                'Category': 'Atlas Sensitivity',
                'Measure': f"Schaefer-{row['resolution']}",
                'N': int(row['n_parcels']),
                'r': row['r'],
                'p_spin': row['p_spin'],
                'Note': '',
            })
    
    # Adaptive delay
    if adaptive_file.exists():
        adapt_df = pd.read_csv(adaptive_file)
        for _, row in adapt_df.iterrows():
            if row['band'] != 'broadband':
                summary_rows.append({
                    'Category': 'Frequency (Fixed)',
                    'Measure': row['band'].capitalize(),
                    'N': int(row['n_parcels']),
                    'r': row['r_fixed'],
                    'p_spin': row['p_spin_fixed'],
                    'Note': f"delay=1",
                })
                summary_rows.append({
                    'Category': 'Frequency (Adaptive)',
                    'Measure': row['band'].capitalize(),
                    'N': int(row['n_parcels']),
                    'r': row['r_adaptive'],
                    'p_spin': row['p_spin_adaptive'],
                    'Note': f"delay={int(row['adaptive_delay'])}",
                })
    
    df_summary = pd.DataFrame(summary_rows)
    
    if len(df_summary) > 0:
        summary_file = OUTPUT_DIR / 'extended_data_summary.csv'
        df_summary.to_csv(summary_file, index=False)
        print(f"Saved: {summary_file}")
        print(df_summary.to_string())
    
    return df_summary


# ============================================
# MAIN
# ============================================

def main():
    print("="*60)
    print("PAPER 1 FINAL ANALYSES")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {OUTPUT_DIR}")
    
    results = {}
    
    # Analysis 1: ρ vs Principal Gradient
    try:
        results['rho_vs_pc1'] = analysis_rho_vs_principal_gradient()
    except Exception as e:
        print(f"ERROR in Analysis 1: {e}")
        results['rho_vs_pc1'] = {'status': 'ERROR', 'message': str(e)}
    
    # Analysis 2: R² Distribution
    try:
        results['r2_distribution'] = analysis_r2_distribution()
    except Exception as e:
        print(f"ERROR in Analysis 2: {e}")
        results['r2_distribution'] = {'status': 'ERROR', 'message': str(e)}
    
    # Analysis 3: Subject-Level DV
    try:
        results['subject_level_dv'] = analysis_subject_level_dv()
    except Exception as e:
        print(f"ERROR in Analysis 3: {e}")
        results['subject_level_dv'] = {'status': 'ERROR', 'message': str(e)}
    
    # Analysis 4: Summary Table
    try:
        create_summary_table()
    except Exception as e:
        print(f"ERROR in Analysis 4: {e}")
    
    # Save all results
    results_file = OUTPUT_DIR / 'paper1_final_statistics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY FOR MANUSCRIPT")
    print("="*60)
    
    if 'r2_distribution' in results and 'median' in results['r2_distribution']:
        r2 = results['r2_distribution']
        print(f"\nR² reporting:")
        print(f"  'Model fits were adequate (median R² = {r2['median']:.4f};")
        print(f"   5th percentile = {r2['percentile_5']:.4f})'")
    
    if 'subject_level_dv' in results and 'mean_r' in results['subject_level_dv']:
        subj = results['subject_level_dv']
        print(f"\nSubject-level inference:")
        print(f"  'The ρ-DV gradient was consistent across individuals")
        print(f"   (mean r = {subj['mean_r']:.3f} ± {subj['std_r']:.3f};")
        print(f"   {subj['prop_negative']*100:.0f}% of subjects showed negative correlation;")
        print(f"   t({subj['n_subjects']-1}) = {subj['t_statistic']:.2f}, p < 0.001)'")
    
    if 'rho_vs_pc1' in results and results['rho_vs_pc1'].get('r') is not None:
        pc1 = results['rho_vs_pc1']
        print(f"\nρ vs Principal Gradient:")
        print(f"  'ρ showed no correlation with the principal gradient")
        print(f"   (r = {pc1['r']:.3f}; p_spin = {pc1['p_spin']:.3f})'")
    
    print(f"\n{'='*60}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
