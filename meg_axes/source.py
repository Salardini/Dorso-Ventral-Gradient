"""
meg_axes/source.py

Subject-specific source modeling and parcellation.

Ensures source reconstruction uses:
    - Subject's FreeSurfer surfaces (SUBJECTS_DIR/sub-XXXX)
    - trans.fif at derivatives/coreg/sub-XXXX/trans.fif
    - BEM solution at derivatives/bem/sub-XXXX/bem-sol.fif
    - Configurable source space spacing (default oct6)
    - Configurable inverse method (default dSPM), pick_ori=normal
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

import mne

from .config import SourceConfig
from .utils import normalize_subject_id


def check_source_prerequisites(
    subject: str,
    subjects_dir: str,
    coreg_dir: str,
    bem_dir: str,
) -> dict:
    """
    Check all prerequisites for source modeling exist.

    Parameters
    ----------
    subject : str
        Subject ID
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    coreg_dir : str
        Directory containing trans.fif files
    bem_dir : str
        Directory containing BEM solutions

    Returns
    -------
    dict
        Status of each prerequisite and paths
    """
    subj = normalize_subject_id(subject)

    fs_dir = os.path.join(subjects_dir, subj)
    trans_path = os.path.join(coreg_dir, subj, "trans.fif")
    bem_path = os.path.join(bem_dir, subj, "bem-sol.fif")

    status = {
        "subject": subj,
        "freesurfer_dir": fs_dir,
        "trans_path": trans_path,
        "bem_path": bem_path,
        "freesurfer_exists": os.path.isdir(fs_dir),
        "trans_exists": os.path.exists(trans_path),
        "bem_exists": os.path.exists(bem_path),
    }

    # Check critical FreeSurfer outputs
    if status["freesurfer_exists"]:
        status["lh_pial"] = os.path.exists(os.path.join(fs_dir, "surf", "lh.pial"))
        status["rh_pial"] = os.path.exists(os.path.join(fs_dir, "surf", "rh.pial"))
        status["lh_white"] = os.path.exists(os.path.join(fs_dir, "surf", "lh.white"))
        status["rh_white"] = os.path.exists(os.path.join(fs_dir, "surf", "rh.white"))

    status["all_ok"] = all([
        status["freesurfer_exists"],
        status["trans_exists"],
        status["bem_exists"],
        status.get("lh_pial", False),
        status.get("rh_pial", False),
    ])

    return status


def get_prerequisite_error_message(status: dict) -> str:
    """
    Generate clear error message for missing prerequisites.

    Parameters
    ----------
    status : dict
        Output from check_source_prerequisites

    Returns
    -------
    str
        Formatted error message with remediation steps
    """
    errors = []
    subj = status["subject"]

    if not status["freesurfer_exists"]:
        errors.append(
            f"ERROR: FreeSurfer reconstruction not found\n"
            f"  Expected: {status['freesurfer_dir']}\n"
            f"  Run: ./scripts/01_reconall.sh {subj}"
        )
    elif not status.get("lh_pial", True) or not status.get("rh_pial", True):
        errors.append(
            f"ERROR: FreeSurfer reconstruction incomplete (missing surfaces)\n"
            f"  Check: {os.path.join(status['freesurfer_dir'], 'scripts', 'recon-all.log')}\n"
            f"  May need to re-run: ./scripts/01_reconall.sh {subj}"
        )

    if not status["bem_exists"]:
        errors.append(
            f"ERROR: BEM solution not found\n"
            f"  Expected: {status['bem_path']}\n"
            f"  Run: python scripts/02_make_bem.py {subj} --config config.yaml"
        )

    if not status["trans_exists"]:
        errors.append(
            f"ERROR: Coregistration transform not found\n"
            f"  Expected: {status['trans_path']}\n"
            f"  See: scripts/03_make_trans.md for instructions"
        )

    return "\n\n".join(errors)


def setup_source_space(
    subject: str,
    subjects_dir: str,
    spacing: str = "oct6",
    verbose: bool = False,
) -> mne.SourceSpaces:
    """
    Set up source space for a subject.

    Parameters
    ----------
    subject : str
        Subject ID
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    spacing : str
        Source space spacing ('oct5', 'oct6', 'ico4', 'ico5', etc.)
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.SourceSpaces
        Source space object
    """
    subj = normalize_subject_id(subject)

    src = mne.setup_source_space(
        subj,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=verbose,
    )

    return src


def make_bem_solution(
    subject: str,
    subjects_dir: str,
    conductivity: tuple = (0.3,),
    verbose: bool = False,
) -> mne.bem.ConductorModel:
    """
    Create BEM solution for a subject.

    Parameters
    ----------
    subject : str
        Subject ID
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    conductivity : tuple
        BEM conductivity (single-shell for MEG)
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.bem.ConductorModel
        BEM solution
    """
    subj = normalize_subject_id(subject)

    # Create BEM model
    model = mne.make_bem_model(
        subject=subj,
        subjects_dir=subjects_dir,
        conductivity=conductivity,
        verbose=verbose,
    )

    # Create BEM solution
    bem = mne.make_bem_solution(model, verbose=verbose)

    return bem


def make_forward_solution(
    info: mne.Info,
    trans_path: str,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel,
    verbose: bool = False,
) -> mne.Forward:
    """
    Create forward solution.

    Parameters
    ----------
    info : mne.Info
        MEG info structure
    trans_path : str
        Path to trans.fif coregistration file
    src : mne.SourceSpaces
        Source space
    bem : mne.bem.ConductorModel
        BEM solution
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.Forward
        Forward solution
    """
    fwd = mne.make_forward_solution(
        info,
        trans=trans_path,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        verbose=verbose,
    )

    return fwd


def make_inverse_operator(
    raw: mne.io.BaseRaw,
    fwd: mne.Forward,
    loose: float = 0.2,
    depth: float = 0.8,
    verbose: bool = False,
) -> mne.minimum_norm.InverseOperator:
    """
    Create inverse operator from raw data and forward solution.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data
    fwd : mne.Forward
        Forward solution
    loose : float
        Loose orientation constraint (0=fixed, 1=free)
    depth : float
        Depth weighting
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.minimum_norm.InverseOperator
        Inverse operator
    """
    # Compute noise covariance from raw data
    cov = mne.compute_raw_covariance(raw, method="empirical", verbose=verbose)

    # Create inverse operator
    inv = mne.minimum_norm.make_inverse_operator(
        raw.info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
        verbose=verbose,
    )

    return inv


def apply_inverse_raw(
    raw: mne.io.BaseRaw,
    inv: mne.minimum_norm.InverseOperator,
    method: str = "dSPM",
    snr: float = 3.0,
    pick_ori: str = "normal",
    verbose: bool = False,
) -> mne.SourceEstimate:
    """
    Apply inverse solution to raw data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data
    inv : mne.minimum_norm.InverseOperator
        Inverse operator
    method : str
        Inverse method ('MNE', 'dSPM', 'sLORETA', 'eLORETA')
    snr : float
        Signal-to-noise ratio assumption
    pick_ori : str
        Orientation picking ('normal', 'vector', None)
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.SourceEstimate
        Source time course estimate
    """
    lambda2 = 1.0 / (snr ** 2)

    stc = mne.minimum_norm.apply_inverse_raw(
        raw,
        inv,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
        verbose=verbose,
    )

    return stc


def extract_label_time_courses(
    stc: mne.SourceEstimate,
    labels: List[mne.Label],
    src: mne.SourceSpaces,
    mode: str = "pca_flip",
    verbose: bool = False,
) -> np.ndarray:
    """
    Extract time courses for parcellation labels.

    Parameters
    ----------
    stc : mne.SourceEstimate
        Source estimate
    labels : list of mne.Label
        Parcellation labels
    src : mne.SourceSpaces
        Source space (from inverse operator)
    mode : str
        Extraction mode ('mean', 'pca_flip', 'mean_flip', etc.)
    verbose : bool
        Verbosity level

    Returns
    -------
    np.ndarray
        (n_labels, n_times) time course matrix
    """
    ts = mne.extract_label_time_course(
        stc,
        labels,
        src,
        mode=mode,
        verbose=verbose,
    )

    return ts.astype(np.float64)


def build_source_model(
    raw: mne.io.BaseRaw,
    subject: str,
    subjects_dir: str,
    trans_path: str,
    bem_path: str,
    config: SourceConfig,
    verbose: bool = False,
) -> mne.minimum_norm.InverseOperator:
    """
    Build complete source model for a subject.

    This is the main entry point that:
    1. Sets up source space
    2. Loads BEM solution
    3. Creates forward solution
    4. Creates inverse operator

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data
    subject : str
        Subject ID
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR
    trans_path : str
        Path to trans.fif
    bem_path : str
        Path to bem-sol.fif
    config : SourceConfig
        Source configuration
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.minimum_norm.InverseOperator
        Ready-to-use inverse operator
    """
    subj = normalize_subject_id(subject)

    # Setup source space
    src = setup_source_space(subj, subjects_dir, config.spacing, verbose)

    # Load BEM solution
    bem = mne.read_bem_solution(bem_path, verbose=verbose)

    # Create forward solution
    fwd = make_forward_solution(raw.info, trans_path, src, bem, verbose)

    # Create inverse operator
    inv = make_inverse_operator(raw, fwd, config.loose, config.depth, verbose)

    return inv


def build_source_model_template(
    raw: mne.io.BaseRaw,
    subjects_dir: str,
    config: SourceConfig,
    verbose: bool = False,
) -> mne.minimum_norm.InverseOperator:
    """
    Build source model using fsaverage template.

    This bypasses subject-specific FreeSurfer reconstruction by using
    the fsaverage template brain for all subjects. Less accurate than
    subject-specific source modeling but doesn't require recon-all.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR (must contain fsaverage)
    config : SourceConfig
        Source configuration
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.minimum_norm.InverseOperator
        Ready-to-use inverse operator
    """
    template = config.template or "fsaverage"
    template_dir = os.path.join(subjects_dir, template)

    # Use pre-computed source space from fsaverage
    src_path = os.path.join(template_dir, "bem", "fsaverage-ico-5-src.fif")
    if not os.path.exists(src_path):
        # Fall back to creating source space
        src = setup_source_space(template, subjects_dir, config.spacing, verbose)
    else:
        src = mne.read_source_spaces(src_path, verbose=verbose)

    # Use pre-computed BEM from fsaverage
    bem_path = os.path.join(template_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
    if not os.path.exists(bem_path):
        raise RuntimeError(
            f"fsaverage BEM not found at {bem_path}. "
            "Run: mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)"
        )
    bem = mne.read_bem_solution(bem_path, verbose=verbose)

    # Create forward solution using fsaverage trans
    # on_inside='ignore' allows processing subjects whose head position
    # causes some sensors to fall inside the template scalp surface
    fwd = mne.make_forward_solution(
        raw.info,
        trans=template,  # MNE handles fsaverage transform automatically
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        on_inside='ignore',
        verbose=verbose,
    )

    # Create inverse operator
    inv = make_inverse_operator(raw, fwd, config.loose, config.depth, verbose)

    return inv


def get_source_summary(inv: mne.minimum_norm.InverseOperator) -> dict:
    """
    Get summary of source model.

    Parameters
    ----------
    inv : mne.minimum_norm.InverseOperator
        Inverse operator

    Returns
    -------
    dict
        Summary statistics
    """
    src = inv["src"]
    info = inv.get("info", {})
    noise_cov = inv.get("noise_cov", {})

    # Get number of MEG channels from info
    n_meg = len(mne.pick_types(info, meg=True)) if info else 0

    return {
        "n_sources_lh": src[0]["nuse"],
        "n_sources_rh": src[1]["nuse"],
        "n_sources_total": src[0]["nuse"] + src[1]["nuse"],
        "source_spacing": src[0].get("subject_his_id", "unknown"),
        "n_meg_channels": n_meg,
        "noise_cov_rank": noise_cov.get("rank", None) if isinstance(noise_cov, dict) else None,
    }
