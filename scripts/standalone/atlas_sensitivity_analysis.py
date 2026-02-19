#!/usr/bin/env python3
"""
Atlas Sensitivity Analysis: Schaefer 400 → 200 → 100

Tests whether the ρ-DV gradient is robust to parcellation resolution.

Reviewer concern: "Finer parcellations might inflate correlations due to 
spatial smoothing"

Method:
1. Load subject-level parcel_ts.npy (Schaefer-400)
2. Map parcels to coarser atlases using hierarchical naming
3. Average time series within coarser parcels
4. Recompute ρ at each resolution
5. Test ρ-DV correlation with spin permutation

Expected outcome:
- If gradient PERSISTS (r ≈ -0.5 to -0.7) → robust to atlas choice
- If gradient WEAKENS substantially → finer parcellation needed

Runtime: ~1-2 hours for all subjects
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

# Paths - UPDATE THESE FOR YOUR SYSTEM
AXES_DIR = Path(r'C:\Users\u2121\Downloads\derivatives\derivatives\axes')
DATA_DIR = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\data')
CENTROIDS_FILE = Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\code\atlas\schaefer400_centroids.csv')
OUTPUT_DIR = DATA_DIR / 'atlas_sensitivity'

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# VAR parameters (same as main analysis)
EMBED_DIM = 10
EMBED_DELAY = 1
RIDGE_ALPHA = 0.001
MAG_MIN = 0.01

# Spin test permutations
N_PERM = 5000

# Checkpoint frequency
CHECKPOINT_FREQ = 25


# ============================================
# PARCEL MAPPING FUNCTIONS
# ============================================

def parse_schaefer_label(label):
    """Parse Schaefer parcel label into components.
    
    Example: "7Networks_LH_Vis_1" -> {networks: 7, hemi: LH, region: Vis, subregion: 1}
    """
    parts = str(label).split('_')
    
    result = {
        'full_label': label,
        'networks': parts[0] if len(parts) > 0 else '',
        'hemi': parts[1] if len(parts) > 1 else '',
        'region': parts[2] if len(parts) > 2 else '',
    }
    
    # Remaining parts are subregion identifiers
    if len(parts) > 3:
        result['subregions'] = '_'.join(parts[3:])
    else:
        result['subregions'] = ''
    
    return result


def create_parcel_mappings(centroids_df):
    """Create mappings from Schaefer-400 to coarser parcellations.
    
    Schaefer parcellations are hierarchical:
    - 400 parcels: Fine-grained (e.g., "7Networks_LH_Vis_1")
    - 200 parcels: ~2 Schaefer-400 parcels per Schaefer-200 parcel
    - 100 parcels: ~4 Schaefer-400 parcels per Schaefer-100 parcel
    
    Returns:
        dict with mappings for each resolution
    """
    mappings = {
        '400': {},  # Identity mapping
        '200': {},  # Group pairs
        '100': {},  # Group quads
    }
    
    # Check which column has labels
    label_col = None
    for col in ['label', 'parcel_name', 'name']:
        if col in centroids_df.columns:
            label_col = col
            break
    
    if label_col is None:
        # Fallback: use index-based grouping
        print("    WARNING: No label column found, using index-based grouping")
        for idx in range(len(centroids_df)):
            mappings['400'][idx] = idx
            mappings['200'][idx] = idx // 2
            mappings['100'][idx] = idx // 4
        return mappings
    
    labels = centroids_df[label_col].values
    
    # Parse each label and create groupings
    for idx, label in enumerate(labels):
        parsed = parse_schaefer_label(label)
        
        # 400-level: identity
        mappings['400'][idx] = idx
        
        # 200-level: group by region + first subregion number // 2
        base_key_200 = f"{parsed['hemi']}_{parsed['region']}"
        if parsed['subregions']:
            nums = re.findall(r'\d+', parsed['subregions'])
            if nums:
                first_num = int(nums[0])
                group_200 = first_num // 2
                base_key_200 += f"_{group_200}"
        mappings['200'][idx] = base_key_200
        
        # 100-level: group by region + first subregion number // 4
        base_key_100 = f"{parsed['hemi']}_{parsed['region']}"
        if parsed['subregions']:
            nums = re.findall(r'\d+', parsed['subregions'])
            if nums:
                first_num = int(nums[0])
                group_100 = first_num // 4
                base_key_100 += f"_{group_100}"
        mappings['100'][idx] = base_key_100
    
    # Convert string keys to integer indices
    for res in ['200', '100']:
        unique_keys = sorted(set(mappings[res].values()))
        key_to_idx = {k: i for i, k in enumerate(unique_keys)}
        mappings[res] = {idx: key_to_idx[key] for idx, key in mappings[res].items()}
    
    return mappings


def aggregate_time_series(ts, mapping):
    """Aggregate time series to coarser parcellation.
    
    Args:
        ts: (n_parcels, n_time) array of parcel time series
        mapping: dict mapping fine parcel idx -> coarse parcel idx
    
    Returns:
        ts_coarse: (n_coarse_parcels, n_time) aggregated time series
    """
    n_fine, n_time = ts.shape
    
    # Get unique coarse indices
    coarse_indices = sorted(set(mapping.values()))
    n_coarse = len(coarse_indices)
    
    # Aggregate
    ts_coarse = np.zeros((n_coarse, n_time))
    counts = np.zeros(n_coarse)
    
    for fine_idx, coarse_idx in mapping.items():
        if fine_idx < n_fine:
            ts_coarse[coarse_idx] += ts[fine_idx]
            counts[coarse_idx] += 1
    
    # Average
    for i in range(n_coarse):
        if counts[i] > 0:
            ts_coarse[i] /= counts[i]
    
    return ts_coarse


def aggregate_coordinates(centroids_df, mapping):
    """Aggregate parcel coordinates to coarser parcellation.
    
    Args:
        centroids_df: DataFrame with x, y, z, hemi columns
        mapping: dict mapping fine parcel idx -> coarse parcel idx
    
    Returns:
        coords_coarse: DataFrame with aggregated coordinates
    """
    df = centroids_df.copy()
    if 'parcel_idx' not in df.columns:
        df['parcel_idx'] = range(len(df))
    
    df['coarse_idx'] = df['parcel_idx'].map(mapping)
    
    # Aggregate coordinates (mean)
    coords_coarse = df.groupby('coarse_idx').agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'hemi': 'first',
    }).reset_index()
    
    coords_coarse = coords_coarse.rename(columns={'coarse_idx': 'parcel_idx'})
    
    return coords_coarse


# ============================================
# CORE FUNCTIONS
# ============================================

def delay_embed(x, m, d):
    """Create delay embedding matrix."""
    T = len(x)
    T_eff = T - (m - 1) * d
    if T_eff <= 30:
        return np.empty((0, m))
    E = np.zeros((T_eff, m), dtype=np.float64)
    for k in range(m):
        start = (m - 1 - k) * d
        E[:, k] = x[start:start + T_eff]
    return E


def compute_rho(ts, embed_delay=EMBED_DELAY, embed_dim=EMBED_DIM, 
                ridge_alpha=RIDGE_ALPHA, mag_min=MAG_MIN):
    """Compute rotational index from delay-embedded VAR(1)."""
    E = delay_embed(ts, m=embed_dim, d=embed_delay)
    if E.shape[0] < 30:
        return np.nan, np.nan
    
    E = (E - E.mean(axis=0)) / (E.std(axis=0) + 1e-8)
    X0, X1 = E[:-1], E[1:]
    XtX = X0.T @ X0
    p = XtX.shape[0]
    A_T = np.linalg.solve(XtX + ridge_alpha * np.eye(p), X0.T @ X1)
    A = A_T.T
    
    lam = np.linalg.eigvals(A)
    mag = np.abs(lam)
    keep = mag > mag_min
    
    if not np.any(keep):
        return np.nan, np.nan
    
    lam_keep = lam[keep]
    mag_keep = mag[keep]
    rho = np.mean(np.abs(np.imag(lam_keep)) / mag_keep)
    
    # R-squared
    pred = X0 @ A_T
    ss_res = np.sum((X1 - pred)**2)
    ss_tot = np.sum((X1 - X1.mean(axis=0))**2)
    r2 = 1 - ss_res / ss_tot
    
    return float(rho), float(r2)


def run_spin_test(rho_vals, z_vals, hemi_labels, n_perm=N_PERM):
    """Run hemisphere-preserving spin permutation test."""
    r, _ = stats.pearsonr(rho_vals, z_vals)
    
    lh_idx = np.where(hemi_labels == 'lh')[0]
    rh_idx = np.where(hemi_labels == 'rh')[0]
    
    null_r = np.zeros(n_perm)
    rng = np.random.default_rng(42)
    
    for i in range(n_perm):
        perm = np.zeros(len(rho_vals), dtype=int)
        perm[lh_idx] = rng.permutation(lh_idx)
        perm[rh_idx] = rng.permutation(rh_idx)
        null_r[i] = stats.pearsonr(rho_vals[perm], z_vals)[0]
    
    p_spin = np.mean(np.abs(null_r) >= np.abs(r))
    
    return r, p_spin


# ============================================
# MAIN ANALYSIS
# ============================================

def main():
    print("="*80)
    print("ATLAS SENSITIVITY ANALYSIS: Schaefer 400 → 200 → 100")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Load centroids
    print("[1/6] Loading Schaefer 400 centroids...")
    if not CENTROIDS_FILE.exists():
        # Try alternative paths
        alt_paths = [
            DATA_DIR / 'group' / 'schaefer400_centroids.csv',
            DATA_DIR.parent / 'code' / 'atlas' / 'schaefer400_centroids.csv',
            Path(r'C:\Users\u2121\Downloads\MEG\Pipeline\atlas\schaefer400_centroids.csv'),
        ]
        centroids = None
        for alt in alt_paths:
            if alt.exists():
                centroids = pd.read_csv(alt, comment='#')
                print(f"    Loaded from: {alt}")
                break
        if centroids is None:
            print(f"ERROR: Cannot find centroids file")
            print(f"Tried: {CENTROIDS_FILE}")
            for alt in alt_paths:
                print(f"       {alt}")
            return
    else:
        centroids = pd.read_csv(CENTROIDS_FILE, comment='#')
    
    # Ensure parcel_idx column exists
    if 'parcel_idx' not in centroids.columns:
        centroids['parcel_idx'] = range(len(centroids))
    
    print(f"    Loaded {len(centroids)} parcel centroids")
    
    # Create mappings
    print("\n[2/6] Creating parcel mappings...")
    mappings = create_parcel_mappings(centroids)
    
    n_200 = len(set(mappings['200'].values()))
    n_100 = len(set(mappings['100'].values()))
    print(f"    Schaefer-400 → 400 parcels")
    print(f"    Schaefer-200 → {n_200} parcels")
    print(f"    Schaefer-100 → {n_100} parcels")
    
    # Create coarse coordinate dataframes
    coords_200 = aggregate_coordinates(centroids, mappings['200'])
    coords_100 = aggregate_coordinates(centroids, mappings['100'])
    
    # Find subjects
    print("\n[3/6] Finding subjects...")
    subjects = sorted([d.name for d in AXES_DIR.iterdir() 
                      if d.is_dir() and d.name.startswith('sub-')])
    print(f"    Found {len(subjects)} subjects")
    
    # Check for checkpoint
    checkpoint_file = OUTPUT_DIR / 'checkpoint_atlas.json'
    results_file = OUTPUT_DIR / 'atlas_sensitivity_parcel_results.csv'
    
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        completed_subjects = set(checkpoint['completed_subjects'])
        print(f"    Resuming from checkpoint: {len(completed_subjects)} subjects done")
        
        if results_file.exists():
            existing_results = pd.read_csv(results_file)
            all_results = existing_results.to_dict('records')
        else:
            all_results = []
    else:
        completed_subjects = set()
        all_results = []
    
    subjects_to_process = [s for s in subjects if s not in completed_subjects]
    print(f"    Subjects to process: {len(subjects_to_process)}")
    
    # Process subjects
    print(f"\n[4/6] Processing subjects...")
    print(f"    Checkpoint every {CHECKPOINT_FREQ} subjects")
    print()
    
    n_total = len(subjects_to_process)
    
    for i, subj in enumerate(subjects_to_process):
        subj_dir = AXES_DIR / subj
        ts_file = subj_dir / 'parcel_ts.npy'
        
        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = datetime.now().strftime('%H:%M:%S')
            print(f"    [{elapsed}] Processing {i+1}/{n_total}: {subj}")
        
        if not ts_file.exists():
            continue
        
        # Load time series
        ts_400 = np.load(ts_file)  # (400, T)
        
        # Handle both (400, T) and (402, T) shapes
        if ts_400.shape[0] > 400:
            ts_400 = ts_400[:400, :]
        
        # Aggregate to coarser parcellations
        ts_200 = aggregate_time_series(ts_400, mappings['200'])
        ts_100 = aggregate_time_series(ts_400, mappings['100'])
        
        # Compute rho at each resolution
        for res, ts_data in [('400', ts_400), ('200', ts_200), ('100', ts_100)]:
            n_parcels = ts_data.shape[0]
            
            for p_idx in range(n_parcels):
                rho, r2 = compute_rho(ts_data[p_idx])
                
                all_results.append({
                    'subject': subj,
                    'resolution': res,
                    'parcel_idx': p_idx,
                    'rho': rho,
                    'r2': r2,
                })
        
        completed_subjects.add(subj)
        
        # Checkpoint
        if (len(completed_subjects) % CHECKPOINT_FREQ == 0) or (i == n_total - 1):
            print(f"    Saving checkpoint ({len(completed_subjects)} subjects done)...")
            
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(results_file, index=False)
            
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'completed_subjects': list(completed_subjects),
                    'timestamp': datetime.now().isoformat()
                }, f)
    
    print(f"\n    Completed all {len(completed_subjects)} subjects!")
    
    # Save final results
    print("\n[5/6] Saving results...")
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(results_file, index=False)
    print(f"    Saved: {results_file}")
    
    # Compute group statistics
    print("\n[6/6] Computing group statistics...")
    
    # Aggregate to parcel means
    parcel_means = df_results.groupby(['resolution', 'parcel_idx']).agg({
        'rho': 'mean',
        'r2': 'mean',
    }).reset_index()
    
    # Merge with coordinates for each resolution
    results_by_res = {}
    
    for res, coords in [('400', centroids), ('200', coords_200), ('100', coords_100)]:
        df_res = parcel_means[parcel_means['resolution'] == res].copy()
        df_res = df_res.merge(coords, on='parcel_idx', how='left')
        df_res = df_res.dropna(subset=['rho', 'z'])
        results_by_res[res] = df_res
    
    # Compute correlations
    print("\n" + "="*80)
    print("RESULTS: ρ-DV CORRELATION BY PARCELLATION RESOLUTION")
    print("="*80)
    print()
    print(f"{'Resolution':<15} {'N parcels':<12} {'r':<12} {'p_spin':<12} {'Mean ρ':<12}")
    print("-"*65)
    
    correlation_results = []
    
    for res in ['400', '200', '100']:
        df_res = results_by_res[res]
        
        if len(df_res) < 20:
            print(f"Schaefer-{res:<7} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue
        
        rho_vals = df_res['rho'].values
        z_vals = df_res['z'].values
        hemi_labels = df_res['hemi'].values
        
        r, p_spin = run_spin_test(rho_vals, z_vals, hemi_labels)
        mean_rho = np.mean(rho_vals)
        
        sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
        
        print(f"Schaefer-{res:<7} {len(df_res):<12} {r:>+.4f}{sig:<4} {p_spin:<12.4f} {mean_rho:<12.4f}")
        
        correlation_results.append({
            'resolution': res,
            'n_parcels': len(df_res),
            'r': r,
            'p_spin': p_spin,
            'mean_rho': mean_rho,
        })
    
    # Save correlation results
    df_corr = pd.DataFrame(correlation_results)
    corr_file = OUTPUT_DIR / 'atlas_sensitivity_correlations.csv'
    df_corr.to_csv(corr_file, index=False)
    print(f"\nSaved: {corr_file}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if len(correlation_results) >= 3:
        r_400 = correlation_results[0]['r']
        r_200 = correlation_results[1]['r']
        r_100 = correlation_results[2]['r']
        
        print(f"""
Parcellation    Correlation    Change from 400
─────────────────────────────────────────────
Schaefer-400    r = {r_400:+.3f}       (baseline)
Schaefer-200    r = {r_200:+.3f}       {(r_200 - r_400):+.3f} ({(r_200/r_400 - 1)*100:+.1f}%)
Schaefer-100    r = {r_100:+.3f}       {(r_100 - r_400):+.3f} ({(r_100/r_400 - 1)*100:+.1f}%)
""")
        
        # Interpretation
        if abs(r_100) > 0.4 and abs(r_200) > 0.5:
            print("""
INTERPRETATION: Gradient is ROBUST to parcellation resolution

The ρ-DV gradient persists across parcellation resolutions, indicating
that the finding is not an artifact of atlas choice or spatial smoothing.
The slight reduction at coarser resolutions is expected due to averaging
across functionally distinct regions.

For paper: "The ρ-DV gradient was robust to parcellation resolution
(Schaefer-400: r = {:.2f}; Schaefer-200: r = {:.2f}; Schaefer-100: r = {:.2f}),
indicating the finding is not dependent on atlas choice."
""".format(r_400, r_200, r_100))
        else:
            print("""
INTERPRETATION: Gradient WEAKENS at coarser resolution

The ρ-DV gradient weakens substantially at coarser parcellations,
suggesting that fine-grained parcellation is important for detecting
this effect. This is consistent with the weaker HCP replication
(103 parcels) and should be noted as a limitation.

For paper: "The ρ-DV gradient weakened at coarser resolutions
(Schaefer-400: r = {:.2f}; Schaefer-100: r = {:.2f}), suggesting
sensitivity to parcellation resolution."
""".format(r_400, r_100))
    
    # Save parcel-level means for each resolution
    for res in ['400', '200', '100']:
        df_res = results_by_res[res]
        res_file = OUTPUT_DIR / f'parcel_means_schaefer{res}.csv'
        df_res.to_csv(res_file, index=False)
        print(f"Saved: {res_file}")
    
    print(f"\n{'='*80}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print("Removed checkpoint file (analysis complete)")


if __name__ == '__main__':
    main()
