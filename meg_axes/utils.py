"""
meg_axes/utils.py

Logging, versioning, DONE markers, and file utilities.

Provides:
    - Version info collection (git hash, software versions)
    - Structured logging setup
    - DONE marker and meta.json writing
    - Path utilities
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Version Information
# =============================================================================

def get_git_hash(repo_path: Optional[str] = None) -> str:
    """
    Get current git commit hash.

    Returns 'unknown' if not in a git repository or git not available.
    """
    try:
        cwd = repo_path or os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_git_dirty(repo_path: Optional[str] = None) -> bool:
    """Check if git repository has uncommitted changes."""
    try:
        cwd = repo_path or os.path.dirname(os.path.dirname(__file__))
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def get_freesurfer_version() -> str:
    """Get FreeSurfer version string."""
    try:
        result = subprocess.run(
            ["recon-all", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split("\n"):
                if "freesurfer" in line.lower() or "recon-all" in line.lower():
                    return line.strip()
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    # Try FREESURFER_HOME
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if fs_home:
        version_file = os.path.join(fs_home, "build-stamp.txt")
        if os.path.exists(version_file):
            with open(version_file) as f:
                return f.read().strip()

    return "unknown"


def get_version_info() -> Dict[str, Any]:
    """
    Collect comprehensive version information.

    Returns dictionary with:
        - pipeline_version
        - git_hash, git_dirty
        - python_version
        - mne_version
        - numpy_version
        - scipy_version
        - freesurfer_version
        - timestamp
    """
    versions = {
        "pipeline_version": "1.0.0",
        "git_hash": get_git_hash(),
        "git_dirty": get_git_dirty(),
        "python_version": sys.version.split()[0],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # MNE
    try:
        import mne
        versions["mne_version"] = mne.__version__
    except ImportError:
        versions["mne_version"] = "not installed"

    # NumPy
    try:
        import numpy as np
        versions["numpy_version"] = np.__version__
    except ImportError:
        versions["numpy_version"] = "not installed"

    # SciPy
    try:
        import scipy
        versions["scipy_version"] = scipy.__version__
    except ImportError:
        versions["scipy_version"] = "not installed"

    # Pandas
    try:
        import pandas as pd
        versions["pandas_version"] = pd.__version__
    except ImportError:
        versions["pandas_version"] = "not installed"

    # FreeSurfer
    versions["freesurfer_version"] = get_freesurfer_version()

    return versions


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up structured logging.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ or script name)
    log_file : str, optional
        Path to log file. Creates parent directories if needed.
    level : int
        Logging level (default: INFO)
    console : bool
        Whether to also log to console (default: True)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# File Utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def write_done_marker(path: str, content: str = "ok") -> None:
    """
    Write DONE marker file.

    Parameters
    ----------
    path : str
        Path to DONE file (typically <output_dir>/DONE)
    content : str
        Content to write (default: "ok")
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(f"{content}\n")


def check_done_marker(path: str) -> bool:
    """Check if DONE marker exists."""
    return os.path.exists(path)


def write_meta_json(
    path: str,
    stage: str,
    subject: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write meta.json with version info, config, and parameters.

    Parameters
    ----------
    path : str
        Path to meta.json file
    stage : str
        Pipeline stage name (e.g., "04_extract_parcels_and_metrics")
    subject : str, optional
        Subject ID
    config : dict, optional
        Configuration used (will be converted to dict if dataclass)
    parameters : dict, optional
        Additional parameters specific to this run
    results : dict, optional
        Summary results (e.g., mean tau, mean rho)
    """
    meta = {
        "stage": stage,
        "versions": get_version_info(),
    }

    if subject:
        meta["subject"] = subject

    if config:
        # Convert dataclass to dict if needed
        if hasattr(config, "__dataclass_fields__"):
            meta["config"] = _dataclass_to_dict(config)
        else:
            meta["config"] = config

    if parameters:
        meta["parameters"] = parameters

    if results:
        meta["results"] = results

    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Recursively convert dataclass to dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {
            k: _dataclass_to_dict(v)
            for k, v in obj.__dict__.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def load_meta_json(path: str) -> Dict[str, Any]:
    """Load meta.json file."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Subject Utilities
# =============================================================================

def normalize_subject_id(subject: str) -> str:
    """Ensure subject ID has 'sub-' prefix."""
    if subject.startswith("sub-"):
        return subject
    return f"sub-{subject}"


def strip_subject_prefix(subject: str) -> str:
    """Remove 'sub-' prefix from subject ID."""
    if subject.startswith("sub-"):
        return subject[4:]
    return subject


def get_subject_paths(
    subject: str,
    derivatives: str,
) -> Dict[str, str]:
    """
    Get standard paths for a subject.

    Returns dictionary with:
        - freesurfer: FreeSurfer subject directory
        - bem: BEM solution path
        - trans: Coregistration transform path
        - axes: Output directory for metrics
        - done: DONE marker path
        - meta: meta.json path
        - log: Log file path
    """
    subj = normalize_subject_id(subject)
    subjects_dir = os.path.join(derivatives, "freesurfer")

    return {
        "freesurfer": os.path.join(subjects_dir, subj),
        "bem": os.path.join(derivatives, "bem", subj, "bem-sol.fif"),
        "trans": os.path.join(derivatives, "coreg", subj, "trans.fif"),
        "axes": os.path.join(derivatives, "axes", subj),
        "done": os.path.join(derivatives, "axes", subj, "DONE"),
        "meta": os.path.join(derivatives, "axes", subj, "meta.json"),
        "log": os.path.join(derivatives, "logs", "extract", f"{subj}.log"),
    }


def check_subject_prerequisites(
    subject: str,
    subjects_dir: str,
    coreg_dir: str,
    bem_dir: str,
) -> Dict[str, bool]:
    """
    Check that all prerequisites exist for a subject.

    Returns dictionary with existence status for each prerequisite.
    """
    subj = normalize_subject_id(subject)

    checks = {
        "freesurfer": os.path.isdir(os.path.join(subjects_dir, subj)),
        "freesurfer_done": os.path.exists(os.path.join(subjects_dir, subj, "DONE")),
        "trans": os.path.exists(os.path.join(coreg_dir, subj, "trans.fif")),
        "bem": os.path.exists(os.path.join(bem_dir, subj, "bem-sol.fif")),
    }

    # Also check for critical FreeSurfer outputs
    fs_dir = os.path.join(subjects_dir, subj)
    if checks["freesurfer"]:
        checks["lh_pial"] = os.path.exists(os.path.join(fs_dir, "surf", "lh.pial"))
        checks["rh_pial"] = os.path.exists(os.path.join(fs_dir, "surf", "rh.pial"))
        checks["lh_white"] = os.path.exists(os.path.join(fs_dir, "surf", "lh.white"))
        checks["rh_white"] = os.path.exists(os.path.join(fs_dir, "surf", "rh.white"))

    return checks


def format_prerequisite_errors(
    subject: str,
    checks: Dict[str, bool],
    subjects_dir: str,
    coreg_dir: str,
    bem_dir: str,
) -> str:
    """Format clear error message for missing prerequisites."""
    subj = normalize_subject_id(subject)
    errors = []

    if not checks.get("freesurfer", False):
        errors.append(
            f"FreeSurfer subject not found: {os.path.join(subjects_dir, subj)}\n"
            f"  Run: ./scripts/01_reconall.sh {subj}"
        )
    elif not checks.get("lh_pial", True) or not checks.get("rh_pial", True):
        errors.append(
            f"FreeSurfer reconstruction incomplete (missing surfaces)\n"
            f"  Check: {os.path.join(subjects_dir, subj, 'scripts', 'recon-all.log')}"
        )

    if not checks.get("bem", False):
        errors.append(
            f"BEM solution not found: {os.path.join(bem_dir, subj, 'bem-sol.fif')}\n"
            f"  Run: python scripts/02_make_bem.py {subj} --config config.yaml"
        )

    if not checks.get("trans", False):
        errors.append(
            f"Coregistration not found: {os.path.join(coreg_dir, subj, 'trans.fif')}\n"
            f"  See: scripts/03_make_trans.md for instructions"
        )

    return "\n\n".join(errors)
