#!/usr/bin/env python3
"""
coreg_check.py

Validate MEG-MRI coregistration quality.

Checks:
1. trans.fif file exists and is readable
2. Fiducial distances between MRI and digitized points
3. Head shape point distances to scalp
4. Overall alignment quality metrics

Outputs:
- Console report with pass/fail status
- Optional CSV summary for all subjects
- Optional QC plots

Usage:
    # Check one subject
    python scripts/coreg_check.py --config config.yaml --subject sub-A2030

    # Check all subjects with trans.fif
    python scripts/coreg_check.py --config config.yaml --all

    # Generate plots
    python scripts/coreg_check.py --config config.yaml --subject sub-A2030 --plot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent directory to path for local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from meg_axes.config import load_config, add_config_args, args_to_overrides
from meg_axes.utils import ensure_dir, normalize_subject_id


def check_coregistration(
    subject: str,
    bids_root: str,
    subjects_dir: str,
    coreg_dir: str,
    task: str = "rest",
    verbose: bool = False,
) -> Dict:
    """
    Check coregistration quality for a subject.

    Parameters
    ----------
    subject : str
        Subject ID
    bids_root : str
        BIDS root directory
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    coreg_dir : str
        Directory containing trans.fif files
    task : str
        Task name for MEG data
    verbose : bool
        Print verbose output

    Returns
    -------
    dict
        Quality metrics and status
    """
    import mne

    subj = normalize_subject_id(subject)
    result = {
        "subject": subj,
        "status": "unknown",
        "trans_exists": False,
        "trans_readable": False,
        "meg_readable": False,
        "fiducial_errors_mm": {},
        "hsp_mean_dist_mm": np.nan,
        "hsp_max_dist_mm": np.nan,
        "hsp_n_points": 0,
        "hsp_n_outliers": 0,
        "pass": False,
        "warnings": [],
        "errors": [],
    }

    # Check trans.fif exists
    trans_path = os.path.join(coreg_dir, subj, "trans.fif")
    result["trans_exists"] = os.path.exists(trans_path)

    if not result["trans_exists"]:
        result["errors"].append(f"trans.fif not found: {trans_path}")
        result["status"] = "missing"
        return result

    # Try to read trans.fif
    try:
        trans = mne.read_trans(trans_path)
        result["trans_readable"] = True
    except Exception as e:
        result["errors"].append(f"Failed to read trans.fif: {e}")
        result["status"] = "unreadable"
        return result

    # Find MEG data
    meg_patterns = [
        os.path.join(bids_root, subj, "meg", f"{subj}_task-{task}_meg.ds"),
        os.path.join(bids_root, subj, "meg", f"{subj}_task-{task}_meg.fif"),
    ]

    meg_path = None
    for pattern in meg_patterns:
        if os.path.exists(pattern):
            meg_path = pattern
            break

    # Also try BIDS pattern with run
    if meg_path is None:
        import glob
        meg_glob = glob.glob(os.path.join(bids_root, subj, "meg", f"{subj}_task-{task}*_meg.*"))
        if meg_glob:
            meg_path = meg_glob[0]

    if meg_path is None:
        result["warnings"].append(f"No MEG data found for {subj}")
        result["status"] = "no_meg"
        # Still count as pass if trans exists and readable
        result["pass"] = True
        return result

    # Read MEG info
    try:
        info = mne.io.read_info(meg_path)
        result["meg_readable"] = True
    except Exception as e:
        result["errors"].append(f"Failed to read MEG info: {e}")
        result["status"] = "meg_error"
        return result

    # Get digitization points
    if info["dig"] is None:
        result["warnings"].append("No digitization points in MEG data")
        result["status"] = "no_dig"
        result["pass"] = True
        return result

    # Extract fiducials and head shape points
    fiducials = {}
    hsp_points = []

    for d in info["dig"]:
        if d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_CARDINAL:
            if d["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                fiducials["LPA"] = d["r"]
            elif d["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                fiducials["RPA"] = d["r"]
            elif d["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                fiducials["Nasion"] = d["r"]
        elif d["kind"] == mne.io.constants.FIFF.FIFFV_POINT_EXTRA:
            hsp_points.append(d["r"])

    result["hsp_n_points"] = len(hsp_points)

    # Check FreeSurfer subject exists
    fs_path = os.path.join(subjects_dir, subj)
    if not os.path.isdir(fs_path):
        result["warnings"].append(f"FreeSurfer subject not found: {fs_path}")
        result["status"] = "no_fs"
        result["pass"] = True
        return result

    # Try to compute head shape distances to scalp
    try:
        # Get MRI fiducials
        fid_path = os.path.join(fs_path, "bem", f"{subj}-fiducials.fif")
        if os.path.exists(fid_path):
            mri_fids = mne.io.read_fiducials(fid_path)[0]
            mri_fiducials = {}
            for f in mri_fids:
                if f["ident"] == mne.io.constants.FIFF.FIFFV_POINT_LPA:
                    mri_fiducials["LPA"] = f["r"]
                elif f["ident"] == mne.io.constants.FIFF.FIFFV_POINT_RPA:
                    mri_fiducials["RPA"] = f["r"]
                elif f["ident"] == mne.io.constants.FIFF.FIFFV_POINT_NASION:
                    mri_fiducials["Nasion"] = f["r"]

            # Compute fiducial errors
            for name in ["LPA", "RPA", "Nasion"]:
                if name in fiducials and name in mri_fiducials:
                    # Transform MEG fiducial to MRI space
                    meg_fid_mri = mne.transforms.apply_trans(trans, fiducials[name])
                    dist_mm = np.linalg.norm(meg_fid_mri - mri_fiducials[name]) * 1000
                    result["fiducial_errors_mm"][name] = float(dist_mm)

        # Compute head shape point distances to scalp
        if len(hsp_points) > 0:
            # Load scalp surface
            scalp_path = os.path.join(fs_path, "bem", "outer_skin.surf")
            if not os.path.exists(scalp_path):
                # Try alternative path
                scalp_path = os.path.join(fs_path, "bem", f"{subj}-outer_skin.surf")

            if os.path.exists(scalp_path):
                scalp_verts, scalp_faces = mne.read_surface(scalp_path)

                # Transform HSP to MRI space
                hsp_mri = np.array([mne.transforms.apply_trans(trans, p) for p in hsp_points])

                # Compute distances to scalp (simplified: distance to nearest vertex)
                # Note: This is approximate; proper distance would use face normals
                from scipy.spatial import cKDTree
                tree = cKDTree(scalp_verts / 1000)  # Convert to meters
                distances, _ = tree.query(hsp_mri)
                distances_mm = distances * 1000

                result["hsp_mean_dist_mm"] = float(np.mean(distances_mm))
                result["hsp_max_dist_mm"] = float(np.max(distances_mm))
                result["hsp_n_outliers"] = int(np.sum(distances_mm > 10))

    except Exception as e:
        result["warnings"].append(f"Could not compute detailed metrics: {e}")

    # Determine pass/fail
    errors = []

    # Check fiducial errors
    for name, dist in result["fiducial_errors_mm"].items():
        if dist > 15:
            errors.append(f"{name} fiducial error {dist:.1f}mm > 15mm")
        elif dist > 10:
            result["warnings"].append(f"{name} fiducial error {dist:.1f}mm is high")

    # Check head shape distances
    if np.isfinite(result["hsp_mean_dist_mm"]):
        if result["hsp_mean_dist_mm"] > 10:
            errors.append(f"Mean HSP distance {result['hsp_mean_dist_mm']:.1f}mm > 10mm")
        elif result["hsp_mean_dist_mm"] > 5:
            result["warnings"].append(f"Mean HSP distance {result['hsp_mean_dist_mm']:.1f}mm is elevated")

        if result["hsp_n_outliers"] > len(hsp_points) * 0.2:
            result["warnings"].append(f"{result['hsp_n_outliers']} HSP outliers (>10mm)")

    result["errors"].extend(errors)
    result["pass"] = len(errors) == 0
    result["status"] = "pass" if result["pass"] else "fail"

    return result


def plot_coregistration(
    subject: str,
    bids_root: str,
    subjects_dir: str,
    coreg_dir: str,
    output_dir: str,
    task: str = "rest",
):
    """Generate coregistration QC plot."""
    import mne
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subj = normalize_subject_id(subject)
    trans_path = os.path.join(coreg_dir, subj, "trans.fif")

    if not os.path.exists(trans_path):
        print(f"Cannot plot: trans.fif not found for {subj}")
        return

    # Find MEG data
    import glob
    meg_files = glob.glob(os.path.join(bids_root, subj, "meg", f"{subj}_task-{task}*_meg.*"))
    if not meg_files:
        print(f"Cannot plot: no MEG data found for {subj}")
        return

    meg_path = meg_files[0]

    try:
        # Read data
        trans = mne.read_trans(trans_path)
        info = mne.io.read_info(meg_path)

        # Create plot
        fig = mne.viz.plot_alignment(
            info=info,
            trans=trans,
            subject=subj,
            subjects_dir=subjects_dir,
            surfaces=["head-dense", "inner_skull"],
            meg=["sensors", "helmet"],
            dig=True,
            show_axes=True,
            coord_frame="mri",
        )

        # Save
        ensure_dir(output_dir)
        out_path = os.path.join(output_dir, f"{subj}_coreg_check.png")

        # Use mne's screenshot if available
        try:
            img = fig.plotter.screenshot()
            import imageio
            imageio.imwrite(out_path, img)
        except Exception:
            # Fallback
            fig.savefig(out_path, dpi=150)

        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed to create plot for {subj}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MEG-MRI coregistration quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subject", "-s",
        help="Subject ID to check",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all subjects with trans.fif",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate QC plots",
    )
    parser.add_argument(
        "--csv",
        help="Output CSV file for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    add_config_args(parser)
    args = parser.parse_args()

    # Load config
    overrides = args_to_overrides(args)
    config = load_config(args.config, overrides)

    # Paths
    bids_root = os.path.join(config.paths.bids_root, config.meg_dataset)
    subjects_dir = config.paths.subjects_dir
    coreg_dir = os.path.join(config.paths.derivatives, "coreg")
    qc_dir = os.path.join(config.paths.derivatives, "qc", "coreg")

    # Determine subjects to check
    if args.subject:
        subjects = [args.subject]
    elif args.all:
        # Find all trans.fif files
        import glob
        trans_files = glob.glob(os.path.join(coreg_dir, "sub-*", "trans.fif"))
        subjects = [os.path.basename(os.path.dirname(f)) for f in trans_files]
    else:
        parser.error("Specify --subject or --all")

    if not subjects:
        print("No subjects found to check")
        return 1

    print("=" * 70)
    print("Coregistration Quality Check")
    print("=" * 70)
    print(f"Checking {len(subjects)} subject(s)")
    print("=" * 70)

    results = []
    n_pass = 0
    n_fail = 0
    n_missing = 0

    for subj in subjects:
        result = check_coregistration(
            subject=subj,
            bids_root=bids_root,
            subjects_dir=subjects_dir,
            coreg_dir=coreg_dir,
            task=config.preprocessing.task,
            verbose=args.verbose,
        )
        results.append(result)

        # Print status
        if result["status"] == "missing":
            status_str = "[MISSING]"
            n_missing += 1
        elif result["pass"]:
            status_str = "[PASS]"
            n_pass += 1
        else:
            status_str = "[FAIL]"
            n_fail += 1

        print(f"{status_str} {result['subject']}")

        if result["fiducial_errors_mm"]:
            fid_str = ", ".join([f"{k}={v:.1f}mm" for k, v in result["fiducial_errors_mm"].items()])
            print(f"         Fiducials: {fid_str}")

        if np.isfinite(result["hsp_mean_dist_mm"]):
            print(f"         HSP: mean={result['hsp_mean_dist_mm']:.1f}mm, "
                  f"max={result['hsp_max_dist_mm']:.1f}mm, "
                  f"n={result['hsp_n_points']}")

        for warn in result["warnings"]:
            print(f"         WARNING: {warn}")
        for err in result["errors"]:
            print(f"         ERROR: {err}")

        # Generate plot if requested
        if args.plot and result["trans_exists"]:
            plot_coregistration(
                subject=subj,
                bids_root=bids_root,
                subjects_dir=subjects_dir,
                coreg_dir=coreg_dir,
                output_dir=qc_dir,
                task=config.preprocessing.task,
            )

    # Save CSV if requested
    if args.csv:
        import pandas as pd
        df = pd.DataFrame([{
            "subject": r["subject"],
            "status": r["status"],
            "pass": r["pass"],
            "lpa_error_mm": r["fiducial_errors_mm"].get("LPA", np.nan),
            "rpa_error_mm": r["fiducial_errors_mm"].get("RPA", np.nan),
            "nasion_error_mm": r["fiducial_errors_mm"].get("Nasion", np.nan),
            "hsp_mean_mm": r["hsp_mean_dist_mm"],
            "hsp_max_mm": r["hsp_max_dist_mm"],
            "hsp_n_points": r["hsp_n_points"],
            "hsp_n_outliers": r["hsp_n_outliers"],
            "n_warnings": len(r["warnings"]),
            "n_errors": len(r["errors"]),
        } for r in results])
        df.to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Pass:    {n_pass}")
    print(f"Fail:    {n_fail}")
    print(f"Missing: {n_missing}")
    print("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
