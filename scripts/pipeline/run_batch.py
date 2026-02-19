#!/usr/bin/env python3
"""
run_batch.py

Batch runner for MEG axes pipeline with dependency management.

Stages (in dependency order):
1. FreeSurfer recon-all (01_reconall.sh)
2. BEM solution (02_make_bem.py)
3. Extract parcels and metrics (04_extract_parcels_and_metrics.py)

Note: Stage 3 (03_make_trans) is manual coregistration and must be done
      separately using the instructions in 03_make_trans.md.

Features:
- Config-based paths and parameters
- Dependency-aware execution order
- Parallel processing with configurable limits
- Failure tracking (failures.csv)
- Skip subjects with existing DONE markers
- Detailed progress reporting

Usage:
    # Run all stages for all subjects
    python scripts/run_batch.py --config config.yaml --stages all

    # Run only extraction for specific subjects
    python scripts/run_batch.py --config config.yaml --stages extract --subjects A2030,A2031

    # Run BEM stage in parallel
    python scripts/run_batch.py --config config.yaml --stages bem --n-jobs 4

    # Dry run to see what would be executed
    python scripts/run_batch.py --config config.yaml --stages all --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for local imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from meg_axes.config import load_config, add_config_args, args_to_overrides
from meg_axes.utils import ensure_dir, normalize_subject_id


# =============================================================================
# Subject Discovery
# =============================================================================

def discover_subjects_meg(bids_root: str) -> List[str]:
    """Discover subjects from BIDS MEG directory."""
    subjects = []
    if not os.path.isdir(bids_root):
        return subjects

    for name in os.listdir(bids_root):
        if name.startswith("sub-") and os.path.isdir(os.path.join(bids_root, name)):
            subjects.append(name.replace("sub-", ""))

    return sorted(subjects)


def discover_subjects_freesurfer(subjects_dir: str) -> List[str]:
    """Discover subjects with existing FreeSurfer reconstructions."""
    subjects = []
    if not os.path.isdir(subjects_dir):
        return subjects

    for name in os.listdir(subjects_dir):
        if name.startswith("sub-") and os.path.isdir(os.path.join(subjects_dir, name)):
            # Check if recon-all completed
            done_file = os.path.join(subjects_dir, name, "scripts", "recon-all.done")
            if os.path.exists(done_file):
                subjects.append(name.replace("sub-", ""))

    return sorted(subjects)


# =============================================================================
# Stage Checking
# =============================================================================

@dataclass
class SubjectStatus:
    """Status of pipeline stages for a subject."""
    subject: str
    has_meg: bool = False
    has_freesurfer: bool = False
    has_bem: bool = False
    has_trans: bool = False
    has_extract: bool = False


def check_subject_status(
    subject: str,
    bids_root: str,
    subjects_dir: str,
    bem_dir: str,
    coreg_dir: str,
    axes_dir: str,
) -> SubjectStatus:
    """Check completion status of all stages for a subject."""
    subj = normalize_subject_id(subject)

    status = SubjectStatus(subject=subj)

    # MEG data
    meg_path = os.path.join(bids_root, subj, "meg")
    status.has_meg = os.path.isdir(meg_path)

    # FreeSurfer
    fs_done = os.path.join(subjects_dir, subj, "scripts", "recon-all.done")
    status.has_freesurfer = os.path.exists(fs_done)

    # BEM
    bem_done = os.path.join(bem_dir, subj, "bem-sol.fif")
    status.has_bem = os.path.exists(bem_done)

    # Trans (coregistration)
    trans_path = os.path.join(coreg_dir, subj, "trans.fif")
    status.has_trans = os.path.exists(trans_path)

    # Extraction
    extract_done = os.path.join(axes_dir, subj, "DONE")
    status.has_extract = os.path.exists(extract_done)

    return status


# =============================================================================
# Stage Execution
# =============================================================================

def run_command(cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str]:
    """Run a command and return (exit_code, output)."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, str(e)


def run_reconall(
    subject: str,
    bids_root: str,
    subjects_dir: str,
    dry_run: bool = False,
) -> Tuple[int, str]:
    """Run FreeSurfer recon-all for a subject."""
    subj = normalize_subject_id(subject)

    # Find T1w image
    anat_dir = os.path.join(bids_root, subj, "anat")
    t1_candidates = [
        os.path.join(anat_dir, f"{subj}_T1w.nii.gz"),
        os.path.join(anat_dir, f"{subj}_T1w.nii"),
    ]

    t1_path = None
    for candidate in t1_candidates:
        if os.path.exists(candidate):
            t1_path = candidate
            break

    if t1_path is None:
        return -1, f"No T1w image found for {subj}"

    cmd = [
        "recon-all",
        "-subject", subj,
        "-i", t1_path,
        "-all",
        "-sd", subjects_dir,
    ]

    if dry_run:
        return 0, f"[DRY-RUN] {' '.join(cmd)}"

    return run_command(cmd, timeout=72000)  # 20 hour timeout for recon-all


def run_bem(
    subject: str,
    config_path: str,
    dry_run: bool = False,
) -> Tuple[int, str]:
    """Run BEM solution creation for a subject."""
    subj = normalize_subject_id(subject)

    script = os.path.join(script_dir, "02_make_bem.py")
    cmd = [
        "python", script,
        "--config", config_path,
        "--subject", subj,
    ]

    if dry_run:
        return 0, f"[DRY-RUN] {' '.join(cmd)}"

    return run_command(cmd, timeout=3600)  # 1 hour timeout


def run_extract(
    subject: str,
    config_path: str,
    dry_run: bool = False,
) -> Tuple[int, str]:
    """Run parcel extraction for a subject."""
    subj = normalize_subject_id(subject)

    script = os.path.join(script_dir, "04_extract_parcels_and_metrics.py")
    cmd = [
        "python", script,
        "--config", config_path,
        "--subject", subj,
    ]

    if dry_run:
        return 0, f"[DRY-RUN] {' '.join(cmd)}"

    return run_command(cmd, timeout=7200)  # 2 hour timeout


# =============================================================================
# Failure Tracking
# =============================================================================

def load_failures(failures_path: str) -> Dict[str, Dict]:
    """Load existing failures from CSV."""
    failures = {}
    if os.path.exists(failures_path):
        with open(failures_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row['subject']}:{row['stage']}"
                failures[key] = row
    return failures


def save_failure(
    failures_path: str,
    subject: str,
    stage: str,
    exit_code: int,
    message: str,
):
    """Append a failure to the failures CSV."""
    file_exists = os.path.exists(failures_path)

    with open(failures_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "subject", "stage", "exit_code", "message"])
        writer.writerow([
            datetime.now().isoformat(),
            subject,
            stage,
            exit_code,
            message[:500],  # Truncate long messages
        ])


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for MEG axes pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stages",
        default="extract",
        help="Stages to run: all | reconall | bem | extract (comma-separated)",
    )
    parser.add_argument(
        "--subjects",
        default="auto",
        help="Subject IDs: 'auto' to discover, or comma-separated list",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel jobs (overrides config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects with completed stages (default behavior)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if already done",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure (overrides config)",
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

    # Parse stages
    if args.stages.lower() == "all":
        stages = ["reconall", "bem", "extract"]
    else:
        stages = [s.strip().lower() for s in args.stages.split(",")]

    # Get job limits from config or args
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        # Use stage-specific limits
        n_jobs = {
            "reconall": config.batch.max_reconall_jobs,
            "bem": config.batch.max_bem_jobs,
            "extract": config.batch.max_extract_jobs,
        }

    fail_fast = args.fail_fast or config.batch.fail_fast
    skip_existing = not args.force and (args.skip_existing or config.batch.skip_existing)

    # Paths
    bids_root = os.path.join(config.paths.bids_root, config.meg_dataset)
    subjects_dir = config.paths.subjects_dir
    bem_dir = os.path.join(config.paths.derivatives, "bem")
    coreg_dir = os.path.join(config.paths.derivatives, "coreg")
    axes_dir = os.path.join(config.paths.derivatives, "axes")
    failures_path = os.path.join(config.paths.derivatives, "failures.csv")

    # Discover subjects
    if args.subjects.lower() == "auto":
        subjects = discover_subjects_meg(bids_root)
    else:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    if not subjects:
        print("ERROR: No subjects found.")
        print(f"Searched in: {bids_root}")
        sys.exit(1)

    print("=" * 70)
    print("MEG Axes Pipeline - Batch Runner")
    print("=" * 70)
    print(f"Config:      {args.config}")
    print(f"BIDS root:   {bids_root}")
    print(f"Subjects:    {len(subjects)} found")
    print(f"Stages:      {', '.join(stages)}")
    print(f"Skip exist:  {skip_existing}")
    print(f"Fail fast:   {fail_fast}")
    print(f"Dry run:     {args.dry_run}")
    print("=" * 70)

    # Check subject status
    print("\nChecking subject status...")
    statuses = {}
    for subj in subjects:
        statuses[subj] = check_subject_status(
            subj, bids_root, subjects_dir, bem_dir, coreg_dir, axes_dir
        )

    # Report status
    n_done = sum(1 for s in statuses.values() if s.has_extract)
    n_missing_trans = sum(1 for s in statuses.values() if not s.has_trans and s.has_freesurfer)

    print(f"  With MEG data:        {sum(1 for s in statuses.values() if s.has_meg)}")
    print(f"  With FreeSurfer:      {sum(1 for s in statuses.values() if s.has_freesurfer)}")
    print(f"  With BEM:             {sum(1 for s in statuses.values() if s.has_bem)}")
    print(f"  With trans.fif:       {sum(1 for s in statuses.values() if s.has_trans)}")
    print(f"  Extraction complete:  {n_done}")

    if n_missing_trans > 0 and "extract" in stages:
        print(f"\nWARNING: {n_missing_trans} subjects have FreeSurfer but missing trans.fif")
        print("         Run manual coregistration first (see 03_make_trans.md)")

    # Process each stage in dependency order
    total_success = 0
    total_fail = 0
    total_skip = 0

    for stage in stages:
        print(f"\n{'=' * 70}")
        print(f"STAGE: {stage.upper()}")
        print(f"{'=' * 70}")

        # Get subjects to process for this stage
        to_process = []
        for subj in subjects:
            status = statuses[subj]

            # Check dependencies
            if stage == "reconall":
                if skip_existing and status.has_freesurfer:
                    continue
                if not status.has_meg:
                    print(f"  [SKIP] {subj}: no MEG data")
                    total_skip += 1
                    continue
                to_process.append(subj)

            elif stage == "bem":
                if skip_existing and status.has_bem:
                    continue
                if not status.has_freesurfer:
                    print(f"  [SKIP] {subj}: no FreeSurfer reconstruction")
                    total_skip += 1
                    continue
                to_process.append(subj)

            elif stage == "extract":
                if skip_existing and status.has_extract:
                    continue
                if not status.has_freesurfer:
                    print(f"  [SKIP] {subj}: no FreeSurfer reconstruction")
                    total_skip += 1
                    continue
                if not status.has_trans:
                    print(f"  [SKIP] {subj}: no trans.fif (run coregistration first)")
                    total_skip += 1
                    continue
                if not status.has_bem:
                    print(f"  [SKIP] {subj}: no BEM solution (run bem stage first)")
                    total_skip += 1
                    continue
                to_process.append(subj)

        if not to_process:
            print(f"  No subjects to process for {stage}")
            continue

        print(f"  Processing {len(to_process)} subjects...")

        # Get job count for this stage
        jobs = n_jobs if isinstance(n_jobs, int) else n_jobs.get(stage, 1)

        # Define runner function
        def run_subject(subj: str) -> Tuple[str, int, str]:
            if stage == "reconall":
                code, out = run_reconall(subj, bids_root, subjects_dir, args.dry_run)
            elif stage == "bem":
                code, out = run_bem(subj, args.config, args.dry_run)
            elif stage == "extract":
                code, out = run_extract(subj, args.config, args.dry_run)
            else:
                code, out = -1, f"Unknown stage: {stage}"
            return subj, code, out

        # Execute
        failed_subjects = []

        if jobs <= 1:
            # Sequential execution
            for subj in to_process:
                subj_id, code, output = run_subject(subj)
                if code == 0:
                    print(f"  [OK] {subj_id}")
                    total_success += 1
                else:
                    print(f"  [FAIL] {subj_id}: exit code {code}")
                    if args.verbose and output:
                        print(f"    {output[:200]}...")
                    total_fail += 1
                    failed_subjects.append((subj_id, code, output))
                    save_failure(failures_path, subj_id, stage, code, output)
                    if fail_fast:
                        print("  Stopping due to --fail-fast")
                        break
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {executor.submit(run_subject, subj): subj for subj in to_process}

                for future in as_completed(futures):
                    subj_id, code, output = future.result()
                    if code == 0:
                        print(f"  [OK] {subj_id}")
                        total_success += 1
                    else:
                        print(f"  [FAIL] {subj_id}: exit code {code}")
                        if args.verbose and output:
                            print(f"    {output[:200]}...")
                        total_fail += 1
                        failed_subjects.append((subj_id, code, output))
                        save_failure(failures_path, subj_id, stage, code, output)

                        if fail_fast:
                            print("  Stopping due to --fail-fast")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success:  {total_success}")
    print(f"Failed:   {total_fail}")
    print(f"Skipped:  {total_skip}")

    if total_fail > 0:
        print(f"\nFailures logged to: {failures_path}")

    print("=" * 70)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
