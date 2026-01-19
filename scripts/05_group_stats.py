#!/usr/bin/env python3
"""
05_group_stats.py

Group-level analysis of parcel-wise MEG metrics.

Computes:
- Group mean/median maps for tau (intrinsic timescale) and rho (rotational index)
- Correlations with anatomical axes (AP, DV, ML) using spin tests
- Tau-rho correlation (raw and residualized)
- Hierarchical robustness analysis (controlling for coordinates, SNR proxies)
- Optional Schaefer-200 robustness check

Inputs:
    derivatives/axes/sub-*/parcel_metrics.csv (with DONE markers)

Outputs:
    derivatives/group/
        parcel_group_maps.csv      # Group means, medians, SDs
        correlation_stats.csv      # All correlations with p-values
        qc_summary.csv             # QC metrics by subject
        meta.json                  # Config, versions, runtime
        figures/*.png              # Visualizations

Usage:
    python scripts/05_group_stats.py --config config.yaml
    python scripts/05_group_stats.py --config config.yaml --n-perm 5000
    python scripts/05_group_stats.py --config config.yaml --schaefer-200  # robustness check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path for local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from meg_axes.config import load_config, add_config_args, args_to_overrides
from meg_axes.utils import ensure_dir, get_version_info, write_meta_json


# =============================================================================
# Correlation and Statistical Tests
# =============================================================================

def compute_correlation(
    a: np.ndarray,
    b: np.ndarray,
    method: str = "spearman"
) -> Tuple[float, float]:
    """
    Compute correlation and parametric p-value.

    Returns (r, p_parametric). Use spin_test for spatial p-values.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 5:
        return np.nan, np.nan

    if method == "pearson":
        r, p = stats.pearsonr(a[mask], b[mask])
    else:
        r, p = stats.spearmanr(a[mask], b[mask])

    return float(r), float(p)


def spin_test(
    map1: np.ndarray,
    map2: np.ndarray,
    coords: np.ndarray,
    n_perm: int = 1000,
    corr_method: str = "spearman",
    fallback_to_perm: bool = True,
) -> Dict:
    """
    Compute correlation with spatial null using spin permutation test.

    Parameters
    ----------
    map1, map2 : np.ndarray
        Parcel-wise maps to correlate
    coords : np.ndarray
        (n_parcels, 3) coordinates for spin permutation
    n_perm : int
        Number of permutations
    corr_method : str
        'spearman' or 'pearson'
    fallback_to_perm : bool
        Fall back to non-spatial permutation if spin test fails

    Returns
    -------
    dict with keys: r, p_spin, p_param, method, null_mean, null_std
    """
    r_obs, p_param = compute_correlation(map1, map2, corr_method)

    if np.isnan(r_obs):
        return {
            "r": np.nan,
            "p_spin": np.nan,
            "p_param": np.nan,
            "method": "none",
            "null_mean": np.nan,
            "null_std": np.nan,
        }

    try:
        from brainspace.null_models import SpinPermutations

        # Prepare coordinates (normalize to unit sphere for spin)
        xyz = coords.astype(np.float64).copy()
        valid = np.all(np.isfinite(xyz), axis=1)

        if np.sum(valid) < coords.shape[0] * 0.9:
            raise ValueError("Too many invalid coordinates for spin test")

        # Normalize to unit sphere
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        xyz = xyz / (norms + 1e-12)

        # Fit spin permutation model
        sp = SpinPermutations(n_rep=n_perm, random_state=42)
        sp.fit(xyz)

        # Generate null distribution by spinning map2
        map2_null = sp.randomize(map2.reshape(-1, 1))[:, :, 0]

        null_corrs = np.array([
            compute_correlation(map1, map2_null[i], corr_method)[0]
            for i in range(n_perm)
        ], dtype=np.float64)

        # Two-tailed p-value
        p_spin = (np.sum(np.abs(null_corrs) >= abs(r_obs)) + 1) / (n_perm + 1)
        method = "spin"

    except (ImportError, ValueError, Exception) as e:
        if not fallback_to_perm:
            return {
                "r": r_obs,
                "p_spin": np.nan,
                "p_param": p_param,
                "method": f"spin_failed:{str(e)[:30]}",
                "null_mean": np.nan,
                "null_std": np.nan,
            }

        # Fall back to non-spatial permutation
        rng = np.random.default_rng(42)
        null_corrs = np.array([
            compute_correlation(map1, rng.permutation(map2), corr_method)[0]
            for _ in range(n_perm)
        ], dtype=np.float64)

        p_spin = (np.sum(np.abs(null_corrs) >= abs(r_obs)) + 1) / (n_perm + 1)
        method = "perm_nonspatial"

    return {
        "r": r_obs,
        "p_spin": float(p_spin),
        "p_param": p_param,
        "method": method,
        "null_mean": float(np.nanmean(null_corrs)),
        "null_std": float(np.nanstd(null_corrs)),
    }


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualize y on X using OLS regression.

    Returns residuals with NaN where inputs were NaN.
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=np.float64)

    if np.sum(mask) < 10:
        return out

    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    out[mask] = y[mask] - lr.predict(X[mask])

    return out


# =============================================================================
# Data Loading
# =============================================================================

def find_subject_files(axes_dir: str, n_parcels: int = 402) -> List[str]:
    """
    Find all parcel_metrics.csv files with DONE markers.

    Also validates that files have expected number of parcels.
    """
    import glob

    pattern = os.path.join(axes_dir, "sub-*", "parcel_metrics.csv")
    files = sorted(glob.glob(pattern))

    valid_files = []
    for f in files:
        done_marker = os.path.join(os.path.dirname(f), "DONE")
        if not os.path.exists(done_marker):
            continue

        # Validate parcel count
        try:
            df = pd.read_csv(f, nrows=5)
            if len(pd.read_csv(f)) == n_parcels:
                valid_files.append(f)
        except Exception:
            pass

    return valid_files


def load_subject_data(
    files: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load all subject data into matrices.

    Returns
    -------
    group_df : pd.DataFrame
        Reference data from first subject (labels, coords, network)
    metrics_dict : dict
        Dictionary of (n_subjects, n_parcels) arrays for each metric
    subjects : list
        Subject IDs
    """
    if len(files) == 0:
        raise RuntimeError("No valid subject files found")

    # Get reference from first subject
    ref_df = pd.read_csv(files[0])
    n_parcels = len(ref_df)

    # Metrics to extract
    metric_cols = [
        "tau", "rho",
        "tau_exp", "tau_exp_r2",
        "rho_r2",
        "ts_var", "ts_rms",
    ]

    # Initialize storage
    metrics = {col: [] for col in metric_cols}
    subjects = []

    for f in files:
        subj = os.path.basename(os.path.dirname(f))
        subjects.append(subj)

        df = pd.read_csv(f)

        for col in metric_cols:
            if col in df.columns:
                metrics[col].append(df[col].to_numpy())
            else:
                metrics[col].append(np.full(n_parcels, np.nan))

    # Stack into matrices
    metrics_mat = {col: np.vstack(arr) for col, arr in metrics.items()}

    return ref_df, metrics_mat, subjects


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute group statistics for tau and rho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=None,
        help="Number of permutations for spin test (overrides config)",
    )
    parser.add_argument(
        "--corr-method",
        choices=["spearman", "pearson"],
        default=None,
        help="Correlation method (overrides config)",
    )
    parser.add_argument(
        "--schaefer-200",
        action="store_true",
        help="Run robustness analysis with Schaefer-200 instead of 400",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    # Add config arguments
    add_config_args(parser)

    args = parser.parse_args()

    # Load config
    overrides = args_to_overrides(args)
    config = load_config(args.config, overrides)

    # Setup
    start_time = time.time()

    # Determine parcellation
    if args.schaefer_200:
        n_parcels = 200
        suffix = "_schaefer200"
    else:
        n_parcels = config.parcellation.n_parcels
        suffix = ""

    # Get parameters
    n_perm = args.n_perm or config.group_stats.n_permutations
    corr_method = args.corr_method or config.group_stats.corr_method
    fallback = config.group_stats.spin_fallback_to_perm

    # Paths
    axes_dir = os.path.join(config.paths.derivatives, "axes")
    out_dir = os.path.join(config.paths.derivatives, f"group{suffix}")
    fig_dir = os.path.join(out_dir, "figures")

    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    print("=" * 70)
    print("MEG Axes Pipeline - Group Statistics")
    print("=" * 70)
    print(f"Input:        {axes_dir}")
    print(f"Output:       {out_dir}")
    print(f"Parcellation: Schaefer-{n_parcels}")
    print(f"Permutations: {n_perm}")
    print(f"Correlation:  {corr_method}")
    print(f"Spin fallback: {fallback}")
    print("=" * 70)

    # ==========================================================================
    # Load data
    # ==========================================================================
    print("\nLoading subject data...")

    files = find_subject_files(axes_dir, n_parcels)
    if len(files) == 0:
        print(f"ERROR: No valid parcel_metrics.csv files found in {axes_dir}")
        print("Ensure subjects have DONE markers and correct parcel count.")
        sys.exit(1)

    ref_df, metrics, subjects = load_subject_data(files)
    n_subjects = len(subjects)

    print(f"Found {n_subjects} subjects with {n_parcels} parcels each")

    # Extract coordinates (should be fsaverage, same for all subjects)
    coords = ref_df[["x", "y", "z"]].to_numpy()
    labels = ref_df["label"].values

    # ==========================================================================
    # Compute group statistics
    # ==========================================================================
    print("\nComputing group statistics...")

    tau_mean = np.nanmean(metrics["tau"], axis=0)
    tau_median = np.nanmedian(metrics["tau"], axis=0)
    tau_std = np.nanstd(metrics["tau"], axis=0)
    tau_n = np.sum(~np.isnan(metrics["tau"]), axis=0)

    rho_mean = np.nanmean(metrics["rho"], axis=0)
    rho_median = np.nanmedian(metrics["rho"], axis=0)
    rho_std = np.nanstd(metrics["rho"], axis=0)

    # QC metrics
    tau_exp_r2_mean = np.nanmean(metrics["tau_exp_r2"], axis=0)
    rho_r2_mean = np.nanmean(metrics["rho_r2"], axis=0)
    ts_var_mean = np.nanmean(metrics["ts_var"], axis=0)
    ts_rms_mean = np.nanmean(metrics["ts_rms"], axis=0)

    # Save group maps
    group_df = pd.DataFrame({
        "parcel_idx": np.arange(n_parcels),
        "label": labels,
        "hemi": ref_df["hemi"].values if "hemi" in ref_df.columns else "unknown",
        "network": ref_df["network"].values if "network" in ref_df.columns else "unknown",

        # Primary metrics
        "tau_mean": tau_mean,
        "tau_median": tau_median,
        "tau_std": tau_std,
        "rho_mean": rho_mean,
        "rho_median": rho_median,
        "rho_std": rho_std,

        # QC
        "tau_exp_r2_mean": tau_exp_r2_mean,
        "rho_r2_mean": rho_r2_mean,
        "ts_var_mean": ts_var_mean,
        "ts_rms_mean": ts_rms_mean,
        "n_subjects": tau_n,

        # Coordinates (fsaverage)
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })

    group_path = os.path.join(out_dir, "parcel_group_maps.csv")
    group_df.to_csv(group_path, index=False)
    print(f"Saved: {group_path}")

    # ==========================================================================
    # Compute correlations with spin tests
    # ==========================================================================
    print("\nComputing correlations with spin tests...")

    stats_rows = []

    # Axis definitions
    axes = {
        "AP": coords[:, 1],  # y = anterior-posterior
        "DV": coords[:, 2],  # z = dorsal-ventral
        "ML": coords[:, 0],  # x = medial-lateral
    }

    # 1. Tau vs anatomical axes
    print("\n  Tau vs anatomical axes:")
    for axis_name, axis_vec in axes.items():
        result = spin_test(tau_mean, axis_vec, coords, n_perm, corr_method, fallback)
        stats_rows.append({
            "map1": "tau_mean",
            "map2": f"axis_{axis_name}",
            "analysis": "raw",
            **result,
        })
        print(f"    {axis_name}: r={result['r']:.3f}, p_spin={result['p_spin']:.4f} ({result['method']})")

    # 2. Rho vs anatomical axes
    print("\n  Rho vs anatomical axes:")
    for axis_name, axis_vec in axes.items():
        result = spin_test(rho_mean, axis_vec, coords, n_perm, corr_method, fallback)
        stats_rows.append({
            "map1": "rho_mean",
            "map2": f"axis_{axis_name}",
            "analysis": "raw",
            **result,
        })
        print(f"    {axis_name}: r={result['r']:.3f}, p_spin={result['p_spin']:.4f} ({result['method']})")

    # 3. Tau vs Rho (raw)
    print("\n  Tau vs Rho:")
    result = spin_test(tau_mean, rho_mean, coords, n_perm, corr_method, fallback)
    stats_rows.append({
        "map1": "tau_mean",
        "map2": "rho_mean",
        "analysis": "raw",
        **result,
    })
    print(f"    Raw: r={result['r']:.3f}, p_spin={result['p_spin']:.4f} ({result['method']})")

    # 4. Residualized on coordinates (xyz)
    print("\n  Tau vs Rho (residualized on xyz):")
    tau_resid_xyz = residualize(tau_mean, coords)
    rho_resid_xyz = residualize(rho_mean, coords)

    result = spin_test(tau_resid_xyz, rho_resid_xyz, coords, n_perm, corr_method, fallback)
    stats_rows.append({
        "map1": "tau_resid_xyz",
        "map2": "rho_resid_xyz",
        "analysis": "resid_xyz",
        **result,
    })
    print(f"    Resid(xyz): r={result['r']:.3f}, p_spin={result['p_spin']:.4f}")

    # 5. Residualized on coordinates + ts_var
    print("\n  Tau vs Rho (residualized on xyz + ts_var):")
    X_xyz_var = np.column_stack([coords, ts_var_mean])
    tau_resid_xyzvar = residualize(tau_mean, X_xyz_var)
    rho_resid_xyzvar = residualize(rho_mean, X_xyz_var)

    result = spin_test(tau_resid_xyzvar, rho_resid_xyzvar, coords, n_perm, corr_method, fallback)
    stats_rows.append({
        "map1": "tau_resid_xyz_var",
        "map2": "rho_resid_xyz_var",
        "analysis": "resid_xyz_var",
        **result,
    })
    print(f"    Resid(xyz+var): r={result['r']:.3f}, p_spin={result['p_spin']:.4f}")

    # 6. Residualized on coordinates + ts_var + ts_rms
    print("\n  Tau vs Rho (residualized on xyz + ts_var + ts_rms):")
    X_xyz_var_rms = np.column_stack([coords, ts_var_mean, ts_rms_mean])
    tau_resid_full = residualize(tau_mean, X_xyz_var_rms)
    rho_resid_full = residualize(rho_mean, X_xyz_var_rms)

    result = spin_test(tau_resid_full, rho_resid_full, coords, n_perm, corr_method, fallback)
    stats_rows.append({
        "map1": "tau_resid_full",
        "map2": "rho_resid_full",
        "analysis": "resid_xyz_var_rms",
        **result,
    })
    print(f"    Resid(full): r={result['r']:.3f}, p_spin={result['p_spin']:.4f}")

    # Save correlation stats
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(out_dir, "correlation_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSaved: {stats_path}")

    # ==========================================================================
    # QC summary by subject
    # ==========================================================================
    print("\nComputing per-subject QC summary...")

    qc_rows = []
    for i, subj in enumerate(subjects):
        qc_rows.append({
            "subject": subj,
            "tau_mean": float(np.nanmean(metrics["tau"][i])),
            "tau_std": float(np.nanstd(metrics["tau"][i])),
            "rho_mean": float(np.nanmean(metrics["rho"][i])),
            "rho_std": float(np.nanstd(metrics["rho"][i])),
            "tau_exp_r2_mean": float(np.nanmean(metrics["tau_exp_r2"][i])),
            "rho_r2_mean": float(np.nanmean(metrics["rho_r2"][i])),
            "ts_var_mean": float(np.nanmean(metrics["ts_var"][i])),
            "n_nan_tau": int(np.sum(np.isnan(metrics["tau"][i]))),
            "n_nan_rho": int(np.sum(np.isnan(metrics["rho"][i]))),
        })

    qc_df = pd.DataFrame(qc_rows)
    qc_path = os.path.join(out_dir, "qc_summary.csv")
    qc_df.to_csv(qc_path, index=False)
    print(f"Saved: {qc_path}")

    # ==========================================================================
    # Generate figures
    # ==========================================================================
    print("\nGenerating figures...")

    plt.style.use("seaborn-v0_8-whitegrid")
    dpi = config.qc.plot_dpi

    # 1. Tau distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tau_mean[np.isfinite(tau_mean)], bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Tau (intrinsic timescale, seconds)")
    ax.set_ylabel("Number of parcels")
    ax.set_title(f"Group mean tau distribution (N={n_subjects})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tau_histogram.png"), dpi=dpi)
    plt.close(fig)

    # 2. Rho distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rho_mean[np.isfinite(rho_mean)], bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Rho (rotational index)")
    ax.set_ylabel("Number of parcels")
    ax.set_title(f"Group mean rho distribution (N={n_subjects})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rho_histogram.png"), dpi=dpi)
    plt.close(fig)

    # 3. Tau vs Rho scatter
    mask = np.isfinite(tau_mean) & np.isfinite(rho_mean)
    r_raw, _ = compute_correlation(tau_mean, rho_mean, corr_method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tau_mean[mask], rho_mean[mask], s=15, alpha=0.6, c="steelblue")
    ax.set_xlabel("Tau (intrinsic timescale)")
    ax.set_ylabel("Rho (rotational index)")
    ax.set_title(f"Tau vs Rho (r={r_raw:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tau_vs_rho.png"), dpi=dpi)
    plt.close(fig)

    # 4. Tau vs Rho residualized
    mask = np.isfinite(tau_resid_xyz) & np.isfinite(rho_resid_xyz)
    r_resid, _ = compute_correlation(tau_resid_xyz, rho_resid_xyz, corr_method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(tau_resid_xyz[mask], rho_resid_xyz[mask], s=15, alpha=0.6, c="coral")
    ax.set_xlabel("Tau (residualized on xyz)")
    ax.set_ylabel("Rho (residualized on xyz)")
    ax.set_title(f"Tau vs Rho residualized (r={r_resid:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tau_vs_rho_residualized.png"), dpi=dpi)
    plt.close(fig)

    # 5. Tau vs AP axis
    mask = np.isfinite(tau_mean) & np.isfinite(coords[:, 1])
    r_ap, _ = compute_correlation(tau_mean, coords[:, 1], corr_method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[mask, 1], tau_mean[mask], s=15, alpha=0.6, c="forestgreen")
    ax.set_xlabel("AP coordinate (y, anterior-posterior)")
    ax.set_ylabel("Tau")
    ax.set_title(f"Tau vs AP axis (r={r_ap:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "tau_vs_AP.png"), dpi=dpi)
    plt.close(fig)

    # 6. Rho vs DV axis
    mask = np.isfinite(rho_mean) & np.isfinite(coords[:, 2])
    r_dv, _ = compute_correlation(rho_mean, coords[:, 2], corr_method)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[mask, 2], rho_mean[mask], s=15, alpha=0.6, c="darkorange")
    ax.set_xlabel("DV coordinate (z, dorsal-ventral)")
    ax.set_ylabel("Rho")
    ax.set_title(f"Rho vs DV axis (r={r_dv:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "rho_vs_DV.png"), dpi=dpi)
    plt.close(fig)

    # 7. QC: tau_exp_r2 distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_r2 = tau_exp_r2_mean[np.isfinite(tau_exp_r2_mean)]
    ax.hist(valid_r2, bins=40, edgecolor="black", alpha=0.7, color="mediumpurple")
    ax.axvline(np.median(valid_r2), color="red", linestyle="--", label=f"median={np.median(valid_r2):.3f}")
    ax.set_xlabel("Tau exponential fit R²")
    ax.set_ylabel("Number of parcels")
    ax.set_title("QC: Tau exponential fit quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qc_tau_exp_r2.png"), dpi=dpi)
    plt.close(fig)

    # 8. QC: rho_r2 distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    valid_r2 = rho_r2_mean[np.isfinite(rho_r2_mean)]
    ax.hist(valid_r2, bins=40, edgecolor="black", alpha=0.7, color="teal")
    ax.axvline(np.median(valid_r2), color="red", linestyle="--", label=f"median={np.median(valid_r2):.3f}")
    ax.set_xlabel("Rho VAR(1) fit R²")
    ax.set_ylabel("Number of parcels")
    ax.set_title("QC: Rho VAR(1) fit quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qc_rho_r2.png"), dpi=dpi)
    plt.close(fig)

    # 9. Split-half reliability (if enough subjects)
    if n_subjects >= 10:
        half1 = np.nanmean(metrics["tau"][:n_subjects//2], axis=0)
        half2 = np.nanmean(metrics["tau"][n_subjects//2:], axis=0)
        mask = np.isfinite(half1) & np.isfinite(half2)
        r_sh, _ = compute_correlation(half1, half2, corr_method)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(half1[mask], half2[mask], s=15, alpha=0.6)
        ax.set_xlabel("Tau (first half subjects)")
        ax.set_ylabel("Tau (second half subjects)")
        ax.set_title(f"Split-half reliability: r={r_sh:.3f}")
        # Add identity line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "tau_split_half_reliability.png"), dpi=dpi)
        plt.close(fig)

    print(f"Figures saved to: {fig_dir}")

    # ==========================================================================
    # Save meta.json
    # ==========================================================================
    elapsed = time.time() - start_time

    meta = {
        "analysis": "group_stats",
        "config_file": args.config,
        "n_subjects": n_subjects,
        "n_parcels": n_parcels,
        "parcellation": f"schaefer{n_parcels}",
        "parameters": {
            "n_permutations": n_perm,
            "corr_method": corr_method,
            "spin_fallback": fallback,
        },
        "subjects": subjects,
        "summary": {
            "tau_mean": float(np.nanmean(tau_mean)),
            "tau_std": float(np.nanstd(tau_mean)),
            "rho_mean": float(np.nanmean(rho_mean)),
            "rho_std": float(np.nanstd(rho_mean)),
            "tau_rho_r": r_raw,
            "tau_rho_r_resid": r_resid,
        },
        "elapsed_s": elapsed,
        "versions": get_version_info(project_root),
    }

    meta_path = os.path.join(out_dir, "meta.json")
    write_meta_json(meta_path, meta)
    print(f"\nSaved: {meta_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Subjects:   {n_subjects}")
    print(f"Parcels:    {n_parcels}")
    print(f"\nTau (intrinsic timescale):")
    print(f"  Mean:   {np.nanmean(tau_mean):.4f} s")
    print(f"  Median: {np.nanmedian(tau_mean):.4f} s")
    print(f"  Std:    {np.nanstd(tau_mean):.4f} s")
    print(f"\nRho (rotational index):")
    print(f"  Mean:   {np.nanmean(rho_mean):.4f}")
    print(f"  Median: {np.nanmedian(rho_mean):.4f}")
    print(f"  Std:    {np.nanstd(rho_mean):.4f}")
    print(f"\nTau-Rho correlation:")
    print(f"  Raw:          r={r_raw:.3f}")
    print(f"  Resid(xyz):   r={r_resid:.3f}")
    print(f"\nElapsed time: {elapsed:.1f}s")
    print("=" * 70)

    print(f"\n[OK] Group statistics saved to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
