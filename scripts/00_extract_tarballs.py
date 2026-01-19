#!/usr/bin/env python3
"""
00_extract_tarballs.py

Extracts subject tar.gz archives to BIDS structure.

The MEG data arrives as tar.gz files:
  - sub-A2002.tar.gz -> extracts to sub-A2002/meg/sub-A2002_task-rest_meg.ds/

This script extracts all tarballs to the working BIDS directory.

Usage:
    # Extract all subjects from source to destination
    python 00_extract_tarballs.py \
        --src /mnt/data/MEG_MOUS \
        --dest /mnt/work/bids/MEG_MOUS

    # Extract specific subjects
    python 00_extract_tarballs.py \
        --src /mnt/data/MEG_MOUS \
        --dest /mnt/work/bids/MEG_MOUS \
        --subjects A2002,A2003,A2004

    # Also extract anatomy tarballs
    python 00_extract_tarballs.py \
        --src /mnt/data/anat \
        --dest /mnt/work/bids/anat \
        --pattern "sub-*.tar.gz"
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tarfile
from typing import List


def ensure_dir(p: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)


def find_tarballs(src_dir: str, pattern: str = "sub-*.tar.gz") -> List[str]:
    """Find all tar.gz files matching pattern."""
    search_pattern = os.path.join(src_dir, pattern)
    tarballs = sorted(glob.glob(search_pattern))

    # Also try without sub- prefix
    if not tarballs:
        alt_pattern = os.path.join(src_dir, "*.tar.gz")
        tarballs = sorted(glob.glob(alt_pattern))

    return tarballs


def get_subject_from_tarball(tarball_path: str) -> str:
    """Extract subject ID from tarball filename."""
    basename = os.path.basename(tarball_path)
    # sub-A2002.tar.gz -> sub-A2002
    # A2002.tar.gz -> A2002
    subject = basename.replace(".tar.gz", "").replace(".tgz", "")
    return subject


def extract_tarball(tarball_path: str, dest_dir: str, overwrite: bool = False) -> bool:
    """
    Extract a single tarball.

    Returns True if extracted, False if skipped.
    """
    subject = get_subject_from_tarball(tarball_path)

    # Normalize subject ID
    subj = subject if subject.startswith("sub-") else f"sub-{subject}"

    # Check if already extracted
    dest_subj_dir = os.path.join(dest_dir, subj)
    done_marker = os.path.join(dest_subj_dir, ".extracted")

    if os.path.exists(done_marker) and not overwrite:
        print(f"[SKIP] {subj}: already extracted")
        return False

    print(f"[EXTRACT] {os.path.basename(tarball_path)} -> {dest_dir}")

    # Remove existing if overwriting
    if os.path.exists(dest_subj_dir) and overwrite:
        shutil.rmtree(dest_subj_dir)

    # Extract
    ensure_dir(dest_dir)

    try:
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Check what's inside - might be sub-A2002/ or just files
            members = tar.getnames()

            # Determine if archive has top-level subject folder
            has_top_folder = any(m.startswith(subj) or m.startswith(subject) for m in members)

            if has_top_folder:
                # Extract directly - archive contains subject folder
                tar.extractall(path=dest_dir)
            else:
                # Extract into subject folder
                ensure_dir(dest_subj_dir)
                tar.extractall(path=dest_subj_dir)

        # Rename if extracted with non-standard name (e.g., A2002 instead of sub-A2002)
        alt_dir = os.path.join(dest_dir, subject)
        if os.path.exists(alt_dir) and not os.path.exists(dest_subj_dir):
            os.rename(alt_dir, dest_subj_dir)

        # Write extraction marker
        with open(done_marker, "w") as f:
            f.write(f"extracted from {tarball_path}\n")

        print(f"[OK] {subj}")
        return True

    except Exception as e:
        print(f"[ERROR] {subj}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract subject tar.gz archives to BIDS structure"
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source directory containing tar.gz files"
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination BIDS directory"
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject IDs (default: all)"
    )
    parser.add_argument(
        "--pattern",
        default="*.tar.gz",
        help="Glob pattern for tarballs (default: *.tar.gz)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extractions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without doing it"
    )

    args = parser.parse_args()

    # Find tarballs
    tarballs = find_tarballs(args.src, args.pattern)

    if not tarballs:
        print(f"No tar.gz files found in {args.src} matching {args.pattern}")
        return

    print(f"Found {len(tarballs)} tar.gz files in {args.src}")

    # Filter by subjects if specified
    if args.subjects:
        subject_list = [s.strip() for s in args.subjects.split(",")]
        # Normalize to match tarball names
        subject_set = set()
        for s in subject_list:
            subject_set.add(s)
            subject_set.add(f"sub-{s}" if not s.startswith("sub-") else s)
            subject_set.add(s.replace("sub-", "") if s.startswith("sub-") else s)

        tarballs = [t for t in tarballs if get_subject_from_tarball(t) in subject_set]
        print(f"Filtered to {len(tarballs)} subjects")

    if args.dry_run:
        print("\n[DRY RUN] Would extract:")
        for t in tarballs:
            subj = get_subject_from_tarball(t)
            print(f"  {os.path.basename(t)} -> {args.dest}/{subj}/")
        return

    # Extract
    ensure_dir(args.dest)

    n_extracted = 0
    n_skipped = 0
    n_failed = 0

    for tarball in tarballs:
        try:
            if extract_tarball(tarball, args.dest, overwrite=args.overwrite):
                n_extracted += 1
            else:
                n_skipped += 1
        except Exception as e:
            print(f"[ERROR] {tarball}: {e}")
            n_failed += 1

    print(f"\nSummary: {n_extracted} extracted, {n_skipped} skipped, {n_failed} failed")


if __name__ == "__main__":
    main()
