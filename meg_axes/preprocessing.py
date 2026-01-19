"""
meg_axes/preprocessing.py

MEG preprocessing with optional ICA artifact removal.

Pipeline:
    1. Load raw MEG data from BIDS
    2. Notch filter (50 or 60 Hz harmonics)
    3. Bandpass filter (l_freq - h_freq)
    4. Resample to target frequency
    5. [Optional] ICA artifact removal
    6. Pick MEG channels only

IMPORTANT: Never compute on a Drive mount. Assume data is staged on local disk.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

from .config import PreprocessingConfig


def find_meg_bids_path(
    bids_root: str,
    subject: str,
    task: str = "rest",
    run: str = "auto",
    session: str = None,
    acquisition: str = None,
) -> BIDSPath:
    """
    Find MEG BIDS path for a subject.

    Parameters
    ----------
    bids_root : str
        Root of BIDS MEG dataset
    subject : str
        Subject ID (with or without 'sub-' prefix)
    task : str
        Task name (default: 'rest')
    run : str
        Run specification:
        - "auto": try without run first, then find first available run
        - "none": no run entity
        - explicit: e.g., "01", "02"
    session : str, optional
        Session name (e.g., 'PeriOp')
    acquisition : str, optional
        Acquisition name (e.g., 'MedOn', 'MedOff')

    Returns
    -------
    BIDSPath
        Valid BIDS path to MEG data

    Raises
    ------
    RuntimeError
        If no valid MEG data found
    """
    subj = subject.replace("sub-", "")

    # Build base kwargs
    kwargs = {
        "root": bids_root,
        "subject": subj,
        "task": task,
        "datatype": "meg",
    }
    if session:
        kwargs["session"] = session
    if acquisition:
        kwargs["acquisition"] = acquisition

    if run and run.lower() == "none":
        return BIDSPath(**kwargs)

    if run is None or run.lower() == "auto":
        # Try without run entity first
        bp = BIDSPath(**kwargs)
        if bp.fpath and os.path.exists(bp.fpath):
            return bp

        # Try available runs
        try:
            runs = [r for r in get_entity_vals(bids_root, "run") if r is not None]
            for r in sorted(runs):
                kwargs["run"] = r
                bp = BIDSPath(**kwargs)
                if bp.fpath and os.path.exists(bp.fpath):
                    return bp
        except Exception:
            pass

        # Try run "1" explicitly
        kwargs["run"] = "1"
        bp = BIDSPath(**kwargs)
        if bp.fpath and os.path.exists(bp.fpath):
            return bp

        raise RuntimeError(
            f"Could not find MEG data for subject={subject}, task={task}, "
            f"session={session}, acquisition={acquisition} in {bids_root}"
        )

    # Explicit run
    kwargs["run"] = run
    return BIDSPath(**kwargs)


def load_raw_meg(
    bids_path: BIDSPath,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """
    Load raw MEG data from BIDS.

    Parameters
    ----------
    bids_path : BIDSPath
        BIDS path to MEG data
    verbose : bool
        Verbosity level

    Returns
    -------
    mne.io.BaseRaw
        Raw MEG data object
    """
    import warnings

    # Try BIDS loading first
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            return read_raw_bids(bids_path, verbose=verbose)
        except Exception:
            pass

        # Try with channel mismatch handling
        try:
            return read_raw_bids(bids_path, verbose=verbose, on_ch_mismatch="reorder")
        except Exception:
            pass

    # Fall back to direct reading
    fpath = str(bids_path.fpath)
    if fpath.endswith(".ds") or os.path.isdir(fpath):
        return mne.io.read_raw_ctf(fpath, preload=False, verbose=verbose)
    elif fpath.endswith(".fif"):
        return mne.io.read_raw_fif(fpath, preload=False, verbose=verbose)
    else:
        raise RuntimeError(f"Could not load MEG data from {bids_path}")


def apply_notch_filter(
    raw: mne.io.BaseRaw,
    freqs: Tuple[float, ...] = (60.0, 120.0, 180.0),
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply notch filter for line noise removal."""
    if freqs:
        raw.notch_filter(list(freqs), verbose=verbose)
    return raw


def apply_bandpass_filter(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Apply bandpass filter."""
    raw.filter(l_freq, h_freq, verbose=verbose)
    return raw


def apply_resample(
    raw: mne.io.BaseRaw,
    sfreq: float = 200.0,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """Resample data to target frequency."""
    raw.resample(sfreq, npad="auto", verbose=verbose)
    return raw


def apply_ica_artifact_removal(
    raw: mne.io.BaseRaw,
    n_components: int = 20,
    method: str = "fastica",
    max_iter: int = 500,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[mne.io.BaseRaw, Optional[mne.preprocessing.ICA]]:
    """Apply ICA for artifact removal (STUB - manual component selection required)."""
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )
    ica.fit(raw, verbose=verbose)

    if ica.exclude:
        raw = ica.apply(raw.copy(), verbose=verbose)

    return raw, ica


def preprocess_raw(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """
    Apply full preprocessing pipeline.

    Pipeline:
        1. Load data into memory
        2. Crop to max duration (if specified)
        3. Notch filter
        4. Bandpass filter
        5. Resample
        6. [Optional] ICA
        7. Pick MEG channels
    """
    # Load data into memory
    raw = raw.copy().load_data()

    # Crop if max duration specified
    if config.max_dur_s is not None:
        max_time = min(raw.times[-1], config.max_dur_s)
        raw.crop(0.0, max_time)

    # Notch filter
    raw = apply_notch_filter(raw, config.notch_freqs, verbose=verbose)

    # Bandpass filter
    raw = apply_bandpass_filter(raw, config.l_freq, config.h_freq, verbose=verbose)

    # Resample
    raw = apply_resample(raw, config.resample_fs, verbose=verbose)

    # Optional ICA
    if config.ica_enabled:
        raw, _ = apply_ica_artifact_removal(
            raw,
            n_components=config.ica_n_components,
            method=config.ica_method,
            max_iter=config.ica_max_iter,
            verbose=verbose,
        )

    # Pick MEG channels only
    raw.pick_types(meg=True, eeg=False, stim=False, eog=False, ecg=False, exclude="bads")

    return raw


def get_preprocessing_summary(raw: mne.io.BaseRaw) -> dict:
    """Get summary of preprocessed data."""
    return {
        "n_channels": len(raw.ch_names),
        "sfreq": raw.info["sfreq"],
        "duration_s": raw.times[-1],
        "n_samples": len(raw.times),
        "channel_types": list(set(raw.get_channel_types())),
        "highpass": raw.info.get("highpass", None),
        "lowpass": raw.info.get("lowpass", None),
    }
