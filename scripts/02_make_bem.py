#!/usr/bin/env python3
"""
02_make_bem.py - Create BEM solution for a single subject

Creates a single-shell BEM model from FreeSurfer reconstruction for MEG
forward modeling.

Usage:
    python 02_make_bem.py sub-A2002
    python 02_make_bem.py sub-A2002 --subjects_dir /path/to/freesurfer
    python 02_make_bem.py sub-A2002 --bem_dir /path/to/bem

Inputs:
    - FreeSurfer reconstruction: $SUBJECTS_DIR/sub-XXX/

Outputs:
    - BEM solution: <bem_dir>/sub-XXX/bem-sol.fif
    - DONE marker: <bem_dir>/sub-XXX/DONE

Environment:
    - SUBJECTS_DIR must be set or passed via --subjects_dir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mne


def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)


def load_config() -> dict:
    """Load configuration from config.yaml if available."""
    import yaml

    config_paths = [
        Path(__file__).parent.parent / "config.yaml",
        Path.cwd() / "config.yaml",
    ]

    for config_path in config_paths:
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)

    return {}


def make_bem_solution(
    subject: str,
    subjects_dir: str,
    bem_dir: str,
    conductivity: tuple = (0.3,),
    overwrite: bool = False,
) -> str:
    """
    Create BEM model and solution for a subject.

    Parameters
    ----------
    subject : str
        Subject ID (e.g., 'sub-A2002')
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    bem_dir : str
        Output directory for BEM solutions
    conductivity : tuple
        BEM conductivity values. Single value (0.3,) for MEG-only single shell.
    overwrite : bool
        If True, overwrite existing BEM solution

    Returns
    -------
    str
        Path to the BEM solution file
    """
    # Ensure subject has sub- prefix
    if not subject.startswith("sub-"):
        subject = f"sub-{subject}"

    # Output paths
    out_dir = os.path.join(bem_dir, subject)
    bem_sol_path = os.path.join(out_dir, "bem-sol.fif")
    done_marker = os.path.join(out_dir, "DONE")

    # Check if already done
    if os.path.exists(done_marker) and not overwrite:
        print(f"[SKIP] {subject}: DONE marker exists at {done_marker}")
        return bem_sol_path

    # Verify FreeSurfer subject exists
    fs_subject_dir = os.path.join(subjects_dir, subject)
    if not os.path.exists(fs_subject_dir):
        raise RuntimeError(
            f"FreeSurfer subject not found: {fs_subject_dir}\n"
            f"Run recon-all first with: ./01_reconall.sh {subject}"
        )

    # Check for required surfaces
    surf_dir = os.path.join(fs_subject_dir, "surf")
    required_surfs = ["lh.pial", "rh.pial", "lh.white", "rh.white"]
    for surf in required_surfs:
        if not os.path.exists(os.path.join(surf_dir, surf)):
            raise RuntimeError(
                f"Missing surface: {surf}\n"
                f"FreeSurfer reconstruction may be incomplete."
            )

    print("=" * 60)
    print(f"Creating BEM solution: {subject}")
    print("=" * 60)
    print(f"SUBJECTS_DIR: {subjects_dir}")
    print(f"Output:       {bem_sol_path}")
    print(f"Conductivity: {conductivity}")
    print("=" * 60)

    # Create output directory
    ensure_dir(out_dir)

    # Step 1: Run watershed BEM if needed
    bem_surfaces_dir = os.path.join(fs_subject_dir, "bem")
    inner_skull = os.path.join(bem_surfaces_dir, "inner_skull.surf")

    if not os.path.exists(inner_skull):
        print("\n[1/3] Running watershed BEM algorithm...")
        mne.bem.make_watershed_bem(
            subject=subject,
            subjects_dir=subjects_dir,
            overwrite=True,
            verbose=True,
        )
    else:
        print("\n[1/3] Watershed BEM surfaces already exist")

    # Step 2: Create BEM model
    print("\n[2/3] Creating BEM model...")
    model = mne.make_bem_model(
        subject=subject,
        subjects_dir=subjects_dir,
        conductivity=conductivity,
        verbose=True,
    )

    # Step 3: Create BEM solution
    print("\n[3/3] Computing BEM solution...")
    bem_sol = mne.make_bem_solution(model, verbose=True)

    # Save
    print(f"\nSaving BEM solution to: {bem_sol_path}")
    mne.write_bem_solution(bem_sol_path, bem_sol, overwrite=True)

    # Write DONE marker
    with open(done_marker, "w") as f:
        f.write("ok\n")

    print(f"\n[OK] {subject} -> {bem_sol_path}")

    return bem_sol_path


def main():
    parser = argparse.ArgumentParser(
        description="Create BEM solution for a subject"
    )
    parser.add_argument(
        "subject",
        help="Subject ID (e.g., sub-A2002 or A2002)"
    )
    parser.add_argument(
        "--subjects_dir",
        default=os.environ.get("SUBJECTS_DIR", "/mnt/work/derivatives/freesurfer"),
        help="FreeSurfer SUBJECTS_DIR"
    )
    parser.add_argument(
        "--bem_dir",
        default="/mnt/work/derivatives/bem",
        help="Output directory for BEM solutions"
    )
    parser.add_argument(
        "--conductivity",
        type=float,
        nargs="+",
        default=[0.3],
        help="BEM conductivity (single value for MEG)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BEM solution"
    )

    args = parser.parse_args()

    # Load config for defaults
    config = load_config()
    if config:
        bem_dir = args.bem_dir
        if bem_dir == "/mnt/work/derivatives/bem":
            bem_dir = config.get("paths", {}).get("derivatives", "/mnt/work/derivatives") + "/bem"
        subjects_dir = args.subjects_dir
        if subjects_dir == "/mnt/work/derivatives/freesurfer":
            subjects_dir = config.get("paths", {}).get("subjects_dir", subjects_dir)
    else:
        bem_dir = args.bem_dir
        subjects_dir = args.subjects_dir

    try:
        make_bem_solution(
            subject=args.subject,
            subjects_dir=subjects_dir,
            bem_dir=bem_dir,
            conductivity=tuple(args.conductivity),
            overwrite=args.overwrite,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
