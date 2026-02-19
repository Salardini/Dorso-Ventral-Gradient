#!/usr/bin/env python3
"""
04_extract_parcels_and_metrics.py

Subject-level MEG source reconstruction and parcel metric extraction.

Pipeline:
1. Load CTF resting-state run from BIDS (task-rest)
2. Preprocess (notch, bandpass, resample, optional ICA)
3. Build subject-specific forward model (requires FreeSurfer + BEM + trans.fif)
4. Inverse solution (dSPM by default, pick_ori=normal)
5. Schaefer-400 parcellation (labels morphed from fsaverage)
6. Extract parcel time series (pca_flip)
7. Compute tau (ACF timescale) and rho (rotational index) with QC metrics
8. Save outputs with meta.json for reproducibility

IMPORTANT:
- Uses fsaverage centroids for coordinates (consistent across subjects)
- Requires: FreeSurfer recon-all, BEM solution, trans.fif coregistration
- Run prerequisite scripts first: 01_reconall.sh, 02_make_bem.py, 03_make_trans

Usage:
    python scripts/04_extract_parcels_and_metrics.py --config config.yaml --subject sub-A2030
    python scripts/04_extract_parcels_and_metrics.py --config config.yaml --subject A2030 --resample 300

Output structure:
    derivatives/axes/sub-XXXX/
        parcel_ts.npy         # (n_parcels, n_time) time series matrix
        parcel_metrics.csv    # tau, rho, QC metrics, fsaverage coords
        meta.json             # config, versions, git hash
        log.txt               # processing log
        DONE                  # completion marker
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import mne

# Add parent directory to path for local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from meg_axes.config import load_config, add_config_args, args_to_overrides
from meg_axes.preprocessing import find_meg_bids_path, load_raw_meg, preprocess_raw
from meg_axes.source import (
    check_source_prerequisites,
    get_prerequisite_error_message,
    build_source_model,
    build_source_model_template,
    apply_inverse_raw,
    extract_label_time_courses,
    get_source_summary,
)
from meg_axes.metrics import (
    compute_tau,
    compute_rho,
    compute_ts_qc,
    preprocess_parcel_ts,
)
from meg_axes.utils import (
    normalize_subject_id,
    get_version_info,
    setup_logging,
    write_done_marker,
    write_meta_json,
    ensure_dir,
)
from atlas.schaefer import (
    get_schaefer_labels,
    get_schaefer_centroids,
    morph_labels_to_subject,
)


def main():
    # =========================================================================
    # Argument parsing
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Extract parcel time series and compute tau/rho metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subject", "-s",
        required=True,
        help="Subject ID (e.g., 'A2030' or 'sub-A2030')",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if DONE marker exists (overrides config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if DONE marker exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # Add standard config arguments
    add_config_args(parser)

    args = parser.parse_args()

    # Load config with CLI overrides
    overrides = args_to_overrides(args)
    config = load_config(args.config, overrides)

    # =========================================================================
    # Setup
    # =========================================================================
    start_time = time.time()
    subj = normalize_subject_id(args.subject)

    # Paths
    bids_root = os.path.join(config.paths.bids_root, config.meg_dataset)
    out_dir = os.path.join(config.paths.derivatives, "axes", subj)
    coreg_dir = os.path.join(config.paths.derivatives, "coreg")
    bem_dir = os.path.join(config.paths.derivatives, "bem")
    subjects_dir = config.paths.subjects_dir
    logs_dir = config.paths.logs_dir

    ensure_dir(out_dir)
    ensure_dir(logs_dir)

    # Setup logging
    log_path = os.path.join(out_dir, "log.txt")
    logger = setup_logging(subj, log_path, level=logging.DEBUG if args.verbose else logging.INFO)

    # Check for existing DONE marker
    done_path = os.path.join(out_dir, "DONE")
    skip_existing = args.skip_existing or config.batch.skip_existing

    if os.path.exists(done_path) and not args.force:
        if skip_existing:
            logger.info(f"SKIP: DONE marker exists at {done_path}")
            return 0
        else:
            logger.warning(f"DONE marker exists but --skip-existing not set, reprocessing")

    logger.info(f"Starting extraction for {subj}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {out_dir}")

    # =========================================================================
    # Check prerequisites (skip for template mode)
    # =========================================================================
    use_template = config.source.template is not None
    trans_path = None
    bem_path = None

    if use_template:
        logger.info(f"Using template-based source modeling ({config.source.template})")
        logger.info("  Skipping subject-specific prerequisite checks")
    else:
        logger.info("Checking source modeling prerequisites...")

        status = check_source_prerequisites(
            subject=subj,
            subjects_dir=subjects_dir,
            coreg_dir=coreg_dir,
            bem_dir=bem_dir,
        )

        if not status["all_ok"]:
            error_msg = get_prerequisite_error_message(status)
            logger.error(f"Prerequisites not met:\n{error_msg}")
            raise RuntimeError(f"Missing prerequisites for {subj}:\n{error_msg}")

        trans_path = status["trans_path"]
        bem_path = status["bem_path"]

        logger.info(f"  FreeSurfer: {status['freesurfer_dir']}")
        logger.info(f"  Trans file: {trans_path}")
        logger.info(f"  BEM solution: {bem_path}")

    # =========================================================================
    # Load and preprocess MEG data
    # =========================================================================
    logger.info("Loading MEG data from BIDS...")

    bids_path = find_meg_bids_path(
        bids_root=bids_root,
        subject=subj,
        task=config.preprocessing.task,
        session=getattr(config.preprocessing, "session", None),
        acquisition=getattr(config.preprocessing, "acquisition", None),
        run=config.preprocessing.run,
    )
    logger.info(f"  BIDS path: {bids_path}")

    raw = load_raw_meg(bids_path, verbose=args.verbose)
    logger.info(f"  Loaded: {raw.info['sfreq']} Hz, {raw.n_times} samples, {len(raw.ch_names)} channels")

    logger.info("Preprocessing...")
    raw = preprocess_raw(raw, config.preprocessing, verbose=args.verbose)
    logger.info(f"  After preprocessing: {raw.info['sfreq']} Hz, {raw.n_times} samples, {len(raw.ch_names)} MEG channels")

    # =========================================================================
    # Build source model
    # =========================================================================
    if use_template:
        logger.info(f"Building template-based source model ({config.source.template})...")
    else:
        logger.info("Building subject-specific source model...")
    logger.info(f"  Spacing: {config.source.spacing}")
    logger.info(f"  Method: {config.source.method}")
    logger.info(f"  SNR: {config.source.snr}")
    logger.info(f"  Loose: {config.source.loose}, Depth: {config.source.depth}")

    if use_template:
        inv = build_source_model_template(
            raw=raw,
            subjects_dir=subjects_dir,
            config=config.source,
            verbose=args.verbose,
        )
    else:
        inv = build_source_model(
            raw=raw,
            subject=subj,
            subjects_dir=subjects_dir,
            trans_path=trans_path,
            bem_path=bem_path,
            config=config.source,
            verbose=args.verbose,
        )

    src_summary = get_source_summary(inv)
    logger.info(f"  Source space: {src_summary['n_sources_total']} sources "
                f"(LH: {src_summary['n_sources_lh']}, RH: {src_summary['n_sources_rh']})")
    logger.info(f"  MEG channels: {src_summary['n_meg_channels']}")

    # =========================================================================
    # Get Schaefer labels and centroids
    # =========================================================================
    logger.info(f"Loading Schaefer-{config.parcellation.n_parcels} parcellation...")

    # Get labels in fsaverage space
    labels_fsavg = get_schaefer_labels(
        n_parcels=config.parcellation.n_parcels,
        n_networks=config.parcellation.n_networks,
        subjects_dir=subjects_dir,
    )
    logger.info(f"  Loaded {len(labels_fsavg)} fsaverage labels")

    # For template mode, use fsaverage labels directly
    # For subject-specific mode, morph labels to subject space
    if use_template:
        logger.info("Using fsaverage labels directly (template mode)")
        labels_subj = labels_fsavg
    else:
        logger.info("Morphing labels to subject space...")
        labels_subj = morph_labels_to_subject(labels_fsavg, subj, subjects_dir)
        logger.info(f"  Morphed {len(labels_subj)} labels to {subj}")

    # Get fsaverage centroids (same for all subjects - critical for group analysis)
    centroids_df, centroids_xyz = get_schaefer_centroids(
        n_parcels=config.parcellation.n_parcels,
        n_networks=config.parcellation.n_networks,
        subjects_dir=subjects_dir,
    )
    logger.info(f"  Using fsaverage centroids from atlas/schaefer{config.parcellation.n_parcels}_centroids.csv")

    # =========================================================================
    # Apply inverse and extract parcel time courses
    # =========================================================================
    logger.info("Applying inverse solution...")

    stc = apply_inverse_raw(
        raw=raw,
        inv=inv,
        method=config.source.method,
        snr=config.source.snr,
        pick_ori="normal",
        verbose=args.verbose,
    )
    logger.info(f"  STC: {stc.data.shape[0]} vertices, {stc.data.shape[1]} time points")

    logger.info(f"Extracting parcel time courses (mode={config.parcellation.extract_mode})...")
    ts_matrix = extract_label_time_courses(
        stc=stc,
        labels=labels_subj,
        src=inv["src"],
        mode=config.parcellation.extract_mode,
        verbose=args.verbose,
    )
    logger.info(f"  Time series matrix: {ts_matrix.shape}")

    n_parcels, n_time = ts_matrix.shape
    fs = raw.info["sfreq"]

    # =========================================================================
    # Compute parcel metrics
    # =========================================================================
    logger.info("Computing parcel metrics...")

    # Initialize result arrays
    tau_integral = np.full(n_parcels, np.nan)
    tau_exp = np.full(n_parcels, np.nan)
    tau_exp_r2 = np.full(n_parcels, np.nan)
    tau_exp_rmse = np.full(n_parcels, np.nan)
    rho = np.full(n_parcels, np.nan)
    rho_r2 = np.full(n_parcels, np.nan)
    rho_n_eig = np.full(n_parcels, np.nan)
    ts_var = np.full(n_parcels, np.nan)
    ts_rms = np.full(n_parcels, np.nan)

    for i in range(n_parcels):
        # Preprocess parcel time series (detrend + standardize)
        ts = preprocess_parcel_ts(ts_matrix[i])

        # QC metrics
        ts_var[i], ts_rms[i] = compute_ts_qc(ts)

        # Tau (intrinsic timescale)
        tau_result = compute_tau(
            ts=ts,
            fs=fs,
            lag_min_s=config.tau.lag_min_s,
            lag_max_s=config.tau.lag_max_s,
        )
        tau_integral[i] = tau_result.tau_integral
        tau_exp[i] = tau_result.tau_exp
        tau_exp_r2[i] = tau_result.tau_exp_r2
        tau_exp_rmse[i] = tau_result.tau_exp_rmse

        # Rho (rotational index)
        rho_result = compute_rho(
            ts=ts,
            embed_dim=config.rho.embed_dim,
            embed_delay=config.rho.embed_delay,
            ridge_alpha=config.rho.ridge_alpha,
            mag_min=config.rho.mag_min,
        )
        rho[i] = rho_result.rho
        rho_r2[i] = rho_result.var1_r2
        rho_n_eig[i] = rho_result.n_eigenvalues_used

    logger.info(f"  Tau (integral): mean={np.nanmean(tau_integral):.4f}, "
                f"std={np.nanstd(tau_integral):.4f}")
    logger.info(f"  Tau (exp fit):  mean={np.nanmean(tau_exp):.4f}, "
                f"mean R²={np.nanmean(tau_exp_r2):.3f}")
    logger.info(f"  Rho:            mean={np.nanmean(rho):.4f}, "
                f"mean R²={np.nanmean(rho_r2):.3f}")

    # =========================================================================
    # Build output DataFrame
    # =========================================================================
    # Use fsaverage centroids for coordinates (critical: same for all subjects)
    df = pd.DataFrame({
        "parcel_idx": np.arange(n_parcels),
        "label": [lb.name for lb in labels_subj],
        "hemi": [lb.hemi for lb in labels_subj],
        "network": centroids_df["network"].values if "network" in centroids_df.columns else "Unknown",

        # Primary metrics
        "tau": tau_integral,
        "rho": rho,

        # Secondary tau estimators
        "tau_exp": tau_exp,
        "tau_exp_r2": tau_exp_r2,
        "tau_exp_rmse": tau_exp_rmse,

        # Rho QC
        "rho_r2": rho_r2,
        "rho_n_eig": rho_n_eig.astype(int),

        # Time series QC
        "ts_var": ts_var,
        "ts_rms": ts_rms,

        # Coordinates (fsaverage, consistent across subjects)
        "x": centroids_xyz[:, 0],  # ML
        "y": centroids_xyz[:, 1],  # AP
        "z": centroids_xyz[:, 2],  # DV
    })

    # =========================================================================
    # Save outputs
    # =========================================================================
    logger.info("Saving outputs...")

    # Time series matrix
    ts_path = os.path.join(out_dir, "parcel_ts.npy")
    np.save(ts_path, ts_matrix)
    logger.info(f"  Saved: {ts_path}")

    # Metrics CSV
    metrics_path = os.path.join(out_dir, "parcel_metrics.csv")
    df.to_csv(metrics_path, index=False)
    logger.info(f"  Saved: {metrics_path}")

    # Meta JSON (reproducibility)
    elapsed = time.time() - start_time
    meta = {
        "subject": subj,
        "bids_path": str(bids_path),
        "config_file": args.config,

        # Data summary
        "n_parcels": n_parcels,
        "n_time": n_time,
        "sfreq": fs,
        "duration_s": n_time / fs,

        # Processing parameters
        "preprocessing": {
            "task": config.preprocessing.task,
            "run": config.preprocessing.run,
            "l_freq": config.preprocessing.l_freq,
            "h_freq": config.preprocessing.h_freq,
            "notch_freqs": list(config.preprocessing.notch_freqs),
            "resample_fs": config.preprocessing.resample_fs,
            "ica_enabled": config.preprocessing.ica_enabled,
        },
        "source": {
            "template": config.source.template,
            "spacing": config.source.spacing,
            "method": config.source.method,
            "snr": config.source.snr,
            "loose": config.source.loose,
            "depth": config.source.depth,
            "n_sources": src_summary["n_sources_total"],
        },
        "parcellation": {
            "atlas": config.parcellation.atlas,
            "n_parcels": config.parcellation.n_parcels,
            "n_networks": config.parcellation.n_networks,
            "extract_mode": config.parcellation.extract_mode,
            "coordinates": "fsaverage_surface",
        },
        "tau": {
            "lag_min_s": config.tau.lag_min_s,
            "lag_max_s": config.tau.lag_max_s,
            "primary_method": config.tau.primary_method,
        },
        "rho": {
            "embed_dim": config.rho.embed_dim,
            "embed_delay": config.rho.embed_delay,
            "ridge_alpha": config.rho.ridge_alpha,
            "mag_min": config.rho.mag_min,
        },

        # Summary statistics
        "metrics_summary": {
            "tau_mean": float(np.nanmean(tau_integral)),
            "tau_std": float(np.nanstd(tau_integral)),
            "tau_exp_mean_r2": float(np.nanmean(tau_exp_r2)),
            "rho_mean": float(np.nanmean(rho)),
            "rho_std": float(np.nanstd(rho)),
            "rho_mean_r2": float(np.nanmean(rho_r2)),
            "ts_var_mean": float(np.nanmean(ts_var)),
        },

        # Reproducibility
        "elapsed_s": elapsed,
        "versions": get_version_info(),
    }

    meta_path = os.path.join(out_dir, "meta.json")
    write_meta_json(meta_path, meta)
    logger.info(f"  Saved: {meta_path}")

    # DONE marker
    write_done_marker(done_path)
    logger.info(f"  Saved: {done_path}")

    logger.info(f"Completed {subj} in {elapsed:.1f}s")
    logger.info(f"Output directory: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
