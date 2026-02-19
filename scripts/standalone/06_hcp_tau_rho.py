#!/usr/bin/env python3
"""
06_hcp_tau_rho.py

Compute tau (intrinsic timescale) and rho (rotational index) on HCP MEG
pre-parcellated resting-state data (BALSA Yeo parcellation).

Input: HCP CIFTI ptseries files ({subject}_MEG_{run}-Restin_bfblpenv_{band}.power.Yeo2011.ptseries.nii)
Output: CSV with tau/rho metrics per parcel, per subject, per band

Usage:
    python scripts/06_hcp_tau_rho.py --input /path/to/hcp_extracted --output /path/to/results
    python scripts/06_hcp_tau_rho.py --input /path/to/hcp_extracted --output /path/to/results --band alpha
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import nibabel as nib

# Add parent directory for local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from meg_axes.metrics import (
    compute_tau,
    compute_rho,
    preprocess_parcel_ts,
    compute_ts_qc,
)


# =============================================================================
# Constants
# =============================================================================

FREQUENCY_BANDS = [
    "delta", "theta", "alpha", "betalow",
    "betahigh", "gammalow", "gammamid", "gammahigh"
]

RUNS = ["3", "4", "5"]  # HCP MEG resting state runs


# =============================================================================
# Data Loading
# =============================================================================

def find_subjects(data_dir: Path) -> List[str]:
    """Find all subject directories."""
    subjects = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.isdigit():
            subjects.append(d.name)
    return subjects


def find_ptseries_files(
    data_dir: Path,
    subject: str,
    band: Optional[str] = None,
) -> List[Path]:
    """Find all ptseries files for a subject."""
    subj_dir = data_dir / subject / "MEG" / "Restin" / "bfblpenv"

    if not subj_dir.exists():
        return []

    pattern = f"*_bfblpenv_{'*' if band is None else band}.power.Yeo2011.ptseries.nii"
    return sorted(subj_dir.glob(pattern))


def load_ptseries(fpath: Path) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load CIFTI ptseries file.

    Returns:
        data: (n_time, n_parcels) array
        sfreq: sampling frequency in Hz
        parcel_names: list of parcel names
    """
    img = nib.load(str(fpath))
    data = np.asarray(img.dataobj)

    # Get sampling frequency from time axis
    ax0 = img.header.get_axis(0)
    sfreq = 1.0 / ax0.step

    # Get parcel names
    ax1 = img.header.get_axis(1)
    parcel_names = [str(n) for n in ax1.name]

    return data, sfreq, parcel_names


def parse_filename(fpath: Path) -> Dict[str, str]:
    """Parse HCP MEG filename to extract metadata."""
    # Example: 100307_MEG_3-Restin_bfblpenv_alpha.power.Yeo2011.ptseries.nii
    name = fpath.name
    parts = name.split("_")

    subject = parts[0]
    run = parts[2].split("-")[0]  # "3-Restin" -> "3"
    band = parts[4].split(".")[0]  # "alpha.power" -> "alpha"

    return {
        "subject": subject,
        "run": run,
        "band": band,
    }


# =============================================================================
# Processing
# =============================================================================

def process_single_file(
    fpath: Path,
    tau_lag_min_s: float = 0.005,
    tau_lag_max_s: float = 0.300,
    embed_dim: int = 10,
    embed_delay: int = 1,
) -> pd.DataFrame:
    """
    Process a single ptseries file to compute tau and rho for all parcels.

    Returns DataFrame with columns:
        subject, run, band, parcel_idx, parcel_name,
        tau, tau_exp, tau_exp_r2, rho, rho_r2, ts_var, ts_rms
    """
    # Load data
    data, sfreq, parcel_names = load_ptseries(fpath)
    n_time, n_parcels = data.shape

    # Parse filename
    meta = parse_filename(fpath)

    # Process each parcel
    results = []
    for i in range(n_parcels):
        ts_raw = data[:, i]

        # Preprocess (detrend + standardize)
        ts = preprocess_parcel_ts(ts_raw)

        # QC metrics
        ts_var, ts_rms = compute_ts_qc(ts)

        # Tau
        tau_result = compute_tau(ts, sfreq, tau_lag_min_s, tau_lag_max_s)

        # Rho
        rho_result = compute_rho(ts, embed_dim, embed_delay)

        results.append({
            "subject": meta["subject"],
            "run": meta["run"],
            "band": meta["band"],
            "parcel_idx": i,
            "parcel_name": parcel_names[i],
            "tau": tau_result.tau_integral,
            "tau_exp": tau_result.tau_exp,
            "tau_exp_r2": tau_result.tau_exp_r2,
            "rho": rho_result.rho,
            "rho_r2": rho_result.var1_r2,
            "ts_var": ts_var,
            "ts_rms": ts_rms,
            "n_time": n_time,
            "sfreq": sfreq,
        })

    return pd.DataFrame(results)


def process_subject(
    data_dir: Path,
    subject: str,
    band: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Process all files for a subject."""
    files = find_ptseries_files(data_dir, subject, band)

    if len(files) == 0:
        if verbose:
            print(f"  No files found for {subject}")
        return pd.DataFrame()

    dfs = []
    for fpath in files:
        if verbose:
            print(f"  Processing {fpath.name}...")

        try:
            df = process_single_file(fpath)
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR processing {fpath.name}: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Group Statistics
# =============================================================================

def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group-level statistics per parcel per band.

    Averages across subjects and runs.
    """
    # Group by band and parcel
    grouped = df.groupby(["band", "parcel_idx", "parcel_name"]).agg({
        "tau": ["mean", "median", "std", "count"],
        "rho": ["mean", "median", "std"],
        "tau_exp_r2": "mean",
        "rho_r2": "mean",
        "ts_var": "mean",
    }).reset_index()

    # Flatten column names
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    return grouped


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute tau/rho on HCP MEG parcellated data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory with extracted HCP data",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--band", "-b",
        choices=FREQUENCY_BANDS,
        default=None,
        help="Process only specific frequency band (default: all)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Process only specific subjects (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    data_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = find_subjects(data_dir)

    print("=" * 70)
    print("HCP MEG Tau/Rho Analysis")
    print("=" * 70)
    print(f"Input:    {data_dir}")
    print(f"Output:   {out_dir}")
    print(f"Subjects: {len(subjects)}")
    print(f"Band:     {args.band or 'all'}")
    print("=" * 70)

    start_time = time.time()

    # Process all subjects
    all_results = []
    for i, subj in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] Processing {subj}...")

        df = process_subject(data_dir, subj, args.band, args.verbose)

        if len(df) > 0:
            all_results.append(df)
            print(f"  Processed {len(df)} parcel-band-run combinations")

    if len(all_results) == 0:
        print("\nERROR: No results generated!")
        return 1

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save subject-level results
    subj_path = out_dir / "hcp_subject_metrics.csv"
    results_df.to_csv(subj_path, index=False)
    print(f"\nSaved: {subj_path}")

    # Compute and save group statistics
    group_df = compute_group_stats(results_df)
    group_path = out_dir / "hcp_group_stats.csv"
    group_df.to_csv(group_path, index=False)
    print(f"Saved: {group_path}")

    # Save metadata
    elapsed = time.time() - start_time
    meta = {
        "analysis": "hcp_tau_rho",
        "n_subjects": len(subjects),
        "n_parcels": results_df["parcel_idx"].nunique(),
        "bands": list(results_df["band"].unique()),
        "runs": list(results_df["run"].unique()),
        "total_observations": len(results_df),
        "elapsed_s": elapsed,
        "subjects": subjects,
    }

    meta_path = out_dir / "hcp_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Subjects processed: {len(subjects)}")
    print(f"Total observations: {len(results_df)}")
    print(f"Parcels: {results_df['parcel_idx'].nunique()}")
    print(f"Bands: {list(results_df['band'].unique())}")
    print(f"\nTau (mean across all): {results_df['tau'].mean():.4f} s")
    print(f"Rho (mean across all): {results_df['rho'].mean():.4f}")
    print(f"\nElapsed time: {elapsed:.1f}s")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
