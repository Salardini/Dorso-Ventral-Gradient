#!/usr/bin/env python3
"""
MOUS Task MEG Analysis: Visual and Auditory tasks
Tests ρ-DV gradient replication and τ-ρ dissociation
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent
HCP_MEG_DIR = Path("G:/My Drive/HCP_MEG")
VISUAL_DIR = HCP_MEG_DIR / "derivatives_visual" / "derivatives_visual" / "axes"
AUDITORY_DIR = HCP_MEG_DIR / "derivatives_auditory" / "derivatives_auditory" / "axes"

OUTPUT_DIR = DATA_DIR / "group"
OUTPUT_DIR.mkdir(exist_ok=True)

N_PERM = 5000

# Reference values from MOUS resting-state
REST_REFERENCE = {
    'rho_z': {'r': -0.73, 'p_spin': 0.002},
    'tau_rho_raw': {'r': -0.02},
    'tau_rho_resid': {'r': -0.62},
}


def load_task_data(task_dir, pattern):
    """Load all subjects for a task."""
    subjects = sorted(task_dir.glob(pattern))
    all_data = []

    for subj_dir in tqdm(subjects, desc=f"Loading {task_dir.parent.name}"):
        csv_file = subj_dir / "parcel_metrics.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            df['subject'] = subj_dir.name
            all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


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


def residualize(y, X):
    """Residualize y with respect to X (single or multiple predictors)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X_design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    return y - X_design @ beta


def analyze_task(df, task_name):
    """Run full analysis for a task."""
    print(f"\n{'='*60}")
    print(f"Analyzing {task_name} task")
    print(f"{'='*60}")

    n_subjects = df['subject'].nunique()
    print(f"Subjects: {n_subjects}")

    # Aggregate to parcel means
    parcel_means = df.groupby(['parcel_idx', 'label', 'hemi']).agg({
        'tau': 'mean',
        'rho': 'mean',
        'tau_exp_r2': 'mean',
        'rho_r2': 'mean',
        'x': 'first',
        'y': 'first',
        'z': 'first',
    }).reset_index()

    parcel_means.columns = ['parcel_idx', 'label', 'hemi', 'tau_mean', 'rho_mean',
                            'tau_r2_mean', 'rho_r2_mean', 'x', 'y', 'z']

    print(f"Parcels: {len(parcel_means)}")

    # Extract arrays
    tau = parcel_means['tau_mean'].values
    rho = parcel_means['rho_mean'].values
    x = parcel_means['x'].values
    y = parcel_means['y'].values
    z = parcel_means['z'].values
    hemi = parcel_means['hemi'].values

    results = {'task': task_name, 'n_subjects': n_subjects, 'n_parcels': len(parcel_means)}

    # Correlation tests
    print(f"\n{'Correlation':<25} {'r':>8} {'p_param':>12} {'p_spin':>10}")
    print("-" * 58)

    # rho vs z (DV)
    r, p_param, p_spin = run_spin_test(rho, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'rho vs z (DV)':<25} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results['rho_z_r'] = r
    results['rho_z_p_spin'] = p_spin

    # tau vs y (AP)
    r, p_param, p_spin = run_spin_test(tau, y, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'tau vs y (AP)':<25} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results['tau_y_r'] = r
    results['tau_y_p_spin'] = p_spin

    # tau vs z (DV)
    r, p_param, p_spin = run_spin_test(tau, z, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'tau vs z (DV)':<25} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results['tau_z_r'] = r
    results['tau_z_p_spin'] = p_spin

    # tau vs rho (raw)
    r, p_param, p_spin = run_spin_test(tau, rho, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'tau vs rho (raw)':<25} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results['tau_rho_raw_r'] = r
    results['tau_rho_raw_p_spin'] = p_spin

    # tau vs rho (residualized for geometry)
    # Residualize both for x, y, z
    coords = np.column_stack([x, y, z])
    tau_resid = residualize(tau, coords)
    rho_resid = residualize(rho, coords)

    r, p_param, p_spin = run_spin_test(tau_resid, rho_resid, hemi)
    sig = '***' if p_spin < 0.001 else '**' if p_spin < 0.01 else '*' if p_spin < 0.05 else ''
    print(f"{'tau vs rho (geom-resid)':<25} {r:>8.4f} {p_param:>12.4e} {p_spin:>10.4f} {sig}")
    results['tau_rho_resid_r'] = r
    results['tau_rho_resid_p_spin'] = p_spin

    # Add residualized values to parcel_means for plotting
    parcel_means['tau_resid'] = tau_resid
    parcel_means['rho_resid'] = rho_resid

    return parcel_means, results


def create_figure(visual_parcels, auditory_parcels, visual_results, auditory_results):
    """Create 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Visual ρ vs DV
    ax = axes[0, 0]
    colors = ['#E24A33' if h == 'lh' else '#348ABD' for h in visual_parcels['hemi']]
    ax.scatter(visual_parcels['z'], visual_parcels['rho_mean'], c=colors, alpha=0.6, s=20)
    ax.set_xlabel('z (DV axis)')
    ax.set_ylabel('ρ (rotational index)')
    r = visual_results['rho_z_r']
    p = visual_results['rho_z_p_spin']
    ax.set_title(f'(A) Visual Task: ρ vs DV\nr = {r:.3f}, p_spin = {p:.4f}')

    # Regression line
    z_sorted = np.sort(visual_parcels['z'])
    slope, intercept = np.polyfit(visual_parcels['z'], visual_parcels['rho_mean'], 1)
    ax.plot(z_sorted, slope * z_sorted + intercept, 'k--', lw=2)

    # Panel B: Auditory ρ vs DV
    ax = axes[0, 1]
    colors = ['#E24A33' if h == 'lh' else '#348ABD' for h in auditory_parcels['hemi']]
    ax.scatter(auditory_parcels['z'], auditory_parcels['rho_mean'], c=colors, alpha=0.6, s=20)
    ax.set_xlabel('z (DV axis)')
    ax.set_ylabel('ρ (rotational index)')
    r = auditory_results['rho_z_r']
    p = auditory_results['rho_z_p_spin']
    ax.set_title(f'(B) Auditory Task: ρ vs DV\nr = {r:.3f}, p_spin = {p:.4f}')

    z_sorted = np.sort(auditory_parcels['z'])
    slope, intercept = np.polyfit(auditory_parcels['z'], auditory_parcels['rho_mean'], 1)
    ax.plot(z_sorted, slope * z_sorted + intercept, 'k--', lw=2)

    # Panel C: Visual τ vs ρ residualized
    ax = axes[1, 0]
    colors = ['#E24A33' if h == 'lh' else '#348ABD' for h in visual_parcels['hemi']]
    ax.scatter(visual_parcels['rho_resid'], visual_parcels['tau_resid'], c=colors, alpha=0.6, s=20)
    ax.set_xlabel('ρ (geometry-residualized)')
    ax.set_ylabel('τ (geometry-residualized)')
    r = visual_results['tau_rho_resid_r']
    p = visual_results['tau_rho_resid_p_spin']
    ax.set_title(f'(C) Visual Task: τ vs ρ (residualized)\nr = {r:.3f}, p_spin = {p:.4f}')

    rho_sorted = np.sort(visual_parcels['rho_resid'])
    slope, intercept = np.polyfit(visual_parcels['rho_resid'], visual_parcels['tau_resid'], 1)
    ax.plot(rho_sorted, slope * rho_sorted + intercept, 'k--', lw=2)

    # Panel D: Auditory τ vs ρ residualized
    ax = axes[1, 1]
    colors = ['#E24A33' if h == 'lh' else '#348ABD' for h in auditory_parcels['hemi']]
    ax.scatter(auditory_parcels['rho_resid'], auditory_parcels['tau_resid'], c=colors, alpha=0.6, s=20)
    ax.set_xlabel('ρ (geometry-residualized)')
    ax.set_ylabel('τ (geometry-residualized)')
    r = auditory_results['tau_rho_resid_r']
    p = auditory_results['tau_rho_resid_p_spin']
    ax.set_title(f'(D) Auditory Task: τ vs ρ (residualized)\nr = {r:.3f}, p_spin = {p:.4f}')

    rho_sorted = np.sort(auditory_parcels['rho_resid'])
    slope, intercept = np.polyfit(auditory_parcels['rho_resid'], auditory_parcels['tau_resid'], 1)
    ax.plot(rho_sorted, slope * rho_sorted + intercept, 'k--', lw=2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#E24A33', markersize=8, label='LH'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#348ABD', markersize=8, label='RH')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    return fig


def main():
    print("=" * 60)
    print("MOUS Task MEG Analysis: Visual & Auditory")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    visual_df = load_task_data(VISUAL_DIR, "sub-V*")
    auditory_df = load_task_data(AUDITORY_DIR, "sub-A*")

    print(f"\nVisual: {visual_df['subject'].nunique()} subjects, {len(visual_df)} rows")
    print(f"Auditory: {auditory_df['subject'].nunique()} subjects, {len(auditory_df)} rows")

    # Analyze each task
    visual_parcels, visual_results = analyze_task(visual_df, "Visual")
    auditory_parcels, auditory_results = analyze_task(auditory_df, "Auditory")

    # Save parcel-level data
    visual_parcels.to_csv(OUTPUT_DIR / "mous_visual_task_group.csv", index=False)
    auditory_parcels.to_csv(OUTPUT_DIR / "mous_auditory_task_group.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'mous_visual_task_group.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'mous_auditory_task_group.csv'}")

    # Save stats comparison
    stats_df = pd.DataFrame([visual_results, auditory_results])

    # Add rest reference
    rest_row = {
        'task': 'Rest (reference)',
        'n_subjects': 203,
        'n_parcels': 400,
        'rho_z_r': REST_REFERENCE['rho_z']['r'],
        'rho_z_p_spin': REST_REFERENCE['rho_z']['p_spin'],
        'tau_rho_raw_r': REST_REFERENCE['tau_rho_raw']['r'],
        'tau_rho_resid_r': REST_REFERENCE['tau_rho_resid']['r'],
    }
    stats_df = pd.concat([stats_df, pd.DataFrame([rest_row])], ignore_index=True)

    stats_df.to_csv(OUTPUT_DIR / "mous_task_stats.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'mous_task_stats.csv'}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Task vs Rest")
    print("=" * 60)
    print(f"\n{'Condition':<15} {'rho-DV r':>10} {'tau-rho raw':>12} {'tau-rho resid':>14}")
    print("-" * 55)
    print(f"{'Rest':<15} {REST_REFERENCE['rho_z']['r']:>10.3f} {REST_REFERENCE['tau_rho_raw']['r']:>12.3f} {REST_REFERENCE['tau_rho_resid']['r']:>14.3f}")
    print(f"{'Visual':<15} {visual_results['rho_z_r']:>10.3f} {visual_results['tau_rho_raw_r']:>12.3f} {visual_results['tau_rho_resid_r']:>14.3f}")
    print(f"{'Auditory':<15} {auditory_results['rho_z_r']:>10.3f} {auditory_results['tau_rho_raw_r']:>12.3f} {auditory_results['tau_rho_resid_r']:>14.3f}")

    # Create figure
    print("\nCreating figure...")
    fig = create_figure(visual_parcels, auditory_parcels, visual_results, auditory_results)
    fig.savefig(OUTPUT_DIR / "mous_task_replication.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'mous_task_replication.png'}")

    plt.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
