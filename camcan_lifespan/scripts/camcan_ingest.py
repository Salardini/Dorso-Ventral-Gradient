#!/usr/bin/env python
"""
camcan_ingest.py — Aim-1 ingestion pipeline for CamCAN resting MEG.

Replicates the published dorsoventral rho-gradient method (meg_axes / MOUS
config) on CamCAN CC700 resting-state MEG, parcel by parcel:

    load raw .fif (MEGIN, raw/unmaxfiltered)
      -> tSSS (Maxwell filter + auto bad-channel detection)   [CamCAN-specific]
      -> notch [50,100,150] -> bandpass 1-40 Hz -> resample 200 Hz
      -> [optional] ICA EOG/ECG removal
      -> fsaverage-template source model (dSPM, no FreeSurfer needed)
      -> Schaefer-400 parcel time courses (pca_flip)
      -> per parcel: detrend+standardize -> rho (VAR(1) delay-embed) + tau
      -> tidy CSV (one row per parcel) + DV-gradient Spearman(rho, z)

Source strategy: no per-subject FreeSurfer. Uses the fsaverage template brain
with MNE's built-in fsaverage head->MRI transform. Coregistration is therefore
template-level (not per-subject scaled) — adequate for a group DV-gradient
replication; per-subject scaled coreg from the head digitization is a future
refinement.

Usage:
    python camcan_ingest.py --limit 1            # smoke test: 1 subject
    python camcan_ingest.py --subject CC110037   # a specific subject
    python camcan_ingest.py --limit 50 --jobs 6  # batch
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# --- make the published pipeline importable (meg_axes, atlas) ---------------
REPO = "/home/salardini/projects/Dorso-Ventral-Gradient"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mne  # noqa: E402
from meg_axes.metrics import compute_rho, compute_tau, compute_ts_qc  # noqa: E402
from meg_axes.source import (  # noqa: E402
    build_source_model_template,
    make_inverse_operator,
    apply_inverse_raw,
    extract_label_time_courses,
)
from meg_axes.config import SourceConfig  # noqa: E402
from mne.coreg import Coregistration  # noqa: E402

from scipy.signal import welch  # noqa: E402

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")


BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}


def band_filter_all(ts2d, fs):
    """Bandpass the full (n_parcels, n_times) array into each canonical band
    (one vectorized filter call per band). Returns {band: filtered 2d array}."""
    out = {}
    for b, (lo, hi) in BANDS.items():
        out[b] = mne.filter.filter_data(ts2d, fs, lo, hi, verbose="ERROR")
    return out


def spectral_features(x, fs):
    """Per-parcel spectral features matching the manuscript:
    spectral_exponent = negative slope of log-log PSD over 2-30 Hz;
    relative band powers vs 1-40 Hz total. x = raw (unstandardized) parcel TS."""
    f, p = welch(x, fs=fs, nperseg=int(2 * fs), noverlap=int(fs))
    m = (f >= 2) & (f <= 30)
    slope = np.polyfit(np.log10(f[m]), np.log10(p[m] + 1e-30), 1)[0]
    tot = p[(f >= 1) & (f <= 40)].sum() + 1e-30
    band = lambda lo, hi: float(p[(f >= lo) & (f < hi)].sum() / tot)
    return dict(spectral_exponent=float(-slope),
                rel_delta=band(1, 4), rel_theta=band(4, 8), rel_alpha=band(8, 13),
                rel_beta=band(13, 30), rel_gamma=band(30, 40))

# --- fixed paths ------------------------------------------------------------
BIDS_BASE = "/home/salardini/data/camcan/raw/cc700/meg/pipeline/release005/BIDSsep"
SUBJECTS_DIR = os.path.expanduser("~/mne_data/MNE-fsaverage-data")
SCHAEFER_PARC = "Schaefer2018_400Parcels_7Networks_order"

# --- published parameters (from config_ds004998.yaml / MOUS) ----------------
PARAMS = dict(
    max_dur_s=300.0,
    notch_freqs=(50.0, 100.0, 150.0),
    l_freq=1.0,
    h_freq=40.0,
    resample_fs=200.0,
    embed_dim=10,
    embed_delay=1,
    ridge_alpha=1e-3,
    mag_min=1e-2,
    inv_method="dSPM",
    snr=3.0,
    extract_mode="pca_flip",
)


def load_schaefer400(subjects_dir):
    """Load Schaefer-400 labels (drop medial wall) + fsaverage pial centroids.
    Returns (labels, centroids_df) with rows aligned to labels order."""
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=SCHAEFER_PARC, subjects_dir=subjects_dir, verbose=False
    )
    drop = ("unknown", "medial_wall", "background", "???")
    labels = [lb for lb in labels if not any(d in lb.name.lower() for d in drop)]
    labels = sorted(labels, key=lambda x: x.name)

    lh, _ = mne.read_surface(os.path.join(subjects_dir, "fsaverage", "surf", "lh.pial"))
    rh, _ = mne.read_surface(os.path.join(subjects_dir, "fsaverage", "surf", "rh.pial"))
    rows = []
    for lb in labels:
        xyz = (lh if lb.hemi == "lh" else rh)[lb.vertices]
        c = xyz.mean(axis=0) if len(lb.vertices) else [np.nan] * 3
        rows.append(dict(label=lb.name, hemi=lb.hemi, x=c[0], y=c[1], z=c[2]))
    return labels, pd.DataFrame(rows)


def preprocess(fif_path, p, do_tsss=True, do_ica=False):
    """Load + clean one CamCAN resting recording -> MEG-only Raw at resample_fs."""
    raw = mne.io.read_raw_fif(fif_path, allow_maxshield=True, preload=True, verbose="ERROR")
    if p["max_dur_s"] is not None:
        raw.crop(0.0, min(raw.times[-1], p["max_dur_s"]))

    if do_tsss:
        # auto-detect flat/noisy channels, then temporal SSS (st_duration=10s).
        # No CamCAN cal/crosstalk files available -> MNE defaults.
        try:
            noisy, flat = mne.preprocessing.find_bad_channels_maxwell(
                raw, coord_frame="head", verbose="ERROR"
            )
            raw.info["bads"] = sorted(set(raw.info["bads"]) | set(noisy) | set(flat))
            raw = mne.preprocessing.maxwell_filter(
                raw, st_duration=10.0, coord_frame="head", verbose="ERROR"
            )
        except Exception as e:
            print(f"    [warn] tSSS failed ({type(e).__name__}: {e}); continuing without")

    raw.notch_filter(list(p["notch_freqs"]), verbose="ERROR")
    raw.filter(p["l_freq"], p["h_freq"], verbose="ERROR")
    raw.resample(p["resample_fs"], npad="auto", verbose="ERROR")

    if do_ica:
        ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                                    max_iter=500, random_state=42, verbose="ERROR")
        ica.fit(raw, verbose="ERROR")
        exclude = []
        try:
            eog_idx, _ = ica.find_bads_eog(raw, verbose="ERROR"); exclude += eog_idx
        except Exception:
            pass
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw, verbose="ERROR"); exclude += ecg_idx
        except Exception:
            pass
        ica.exclude = sorted(set(exclude))
        if ica.exclude:
            ica.apply(raw, verbose="ERROR")

    raw.pick_types(meg=True, eeg=False, stim=False, eog=False, ecg=False, exclude="bads")
    return raw


def build_inverse_coreg(raw, subjects_dir, src_cfg):
    """Per-subject coregistration: fit fsaverage to the head digitization via
    ICP, then build forward+inverse on fsaverage src/BEM with that trans.
    Returns (inv, median_dig_mri_dist_mm). No FreeSurfer recon-all needed."""
    fa = os.path.join(subjects_dir, "fsaverage", "bem")
    src = mne.read_source_spaces(os.path.join(fa, "fsaverage-ico-5-src.fif"), verbose=False)
    bem = mne.read_bem_solution(
        os.path.join(fa, "fsaverage-5120-5120-5120-bem-sol.fif"), verbose=False)

    coreg = Coregistration(raw.info, "fsaverage", subjects_dir, fiducials="estimated")
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)
    coreg.omit_head_shape_points(distance=5e-3)
    coreg.fit_icp(n_iterations=20, verbose=False)

    dists = coreg.compute_dig_mri_distances() * 1e3  # mm
    fwd = mne.make_forward_solution(
        raw.info, trans=coreg.trans, src=src, bem=bem,
        meg=True, eeg=False, mindist=5.0, on_inside="ignore", verbose=False)
    inv = make_inverse_operator(raw, fwd, src_cfg.loose, src_cfg.depth, verbose=False)
    return inv, float(np.median(dists))


def process_subject(fif_path, labels, centroids, p, do_tsss=True, do_ica=False,
                    coreg_mode="auto", do_bands=False):
    """Full per-subject pipeline -> DataFrame (one row per parcel)."""
    subj = os.path.basename(fif_path).split("_")[0].replace("sub-", "")
    t0 = time.time()
    raw = preprocess(fif_path, p, do_tsss=do_tsss, do_ica=do_ica)

    src_cfg = SourceConfig()  # dSPM, loose .2, depth .8
    coreg_dist = np.nan
    if coreg_mode == "auto":
        inv, coreg_dist = build_inverse_coreg(raw, SUBJECTS_DIR, src_cfg)
    else:  # generic fsaverage template trans (the old shortcut)
        inv = build_source_model_template(raw, SUBJECTS_DIR, src_cfg, verbose=False)
    stc = apply_inverse_raw(raw, inv, method=p["inv_method"], snr=p["snr"],
                            pick_ori="normal", verbose=False)
    ts = extract_label_time_courses(stc, labels, inv["src"], mode=p["extract_mode"])
    fs = raw.info["sfreq"]
    band_ts = band_filter_all(np.asarray(ts, dtype=np.float64), fs) if do_bands else None

    rows = []
    for i, lb in enumerate(labels):
        raw_ts = ts[i].astype(np.float64)
        spec = spectral_features(raw_ts, fs)
        band_rho = {}
        if band_ts is not None:
            for b, bt in band_ts.items():
                xb = bt[i]; xb = (xb - xb.mean()) / (xb.std() + 1e-12)
                band_rho[f"rho_{b}"] = compute_rho(
                    xb, embed_dim=p["embed_dim"], embed_delay=p["embed_delay"],
                    ridge_alpha=p["ridge_alpha"], mag_min=p["mag_min"]).rho
        x = (raw_ts - raw_ts.mean())
        x = x / (x.std() + 1e-12)
        rr = compute_rho(x, embed_dim=p["embed_dim"], embed_delay=p["embed_delay"],
                         ridge_alpha=p["ridge_alpha"], mag_min=p["mag_min"])
        tr = compute_tau(x, fs)
        var, rms = compute_ts_qc(x)
        c = centroids.iloc[i]
        rows.append(dict(
            subject=subj, parcel_idx=i, label=lb.name, hemi=lb.hemi,
            x=c["x"], y=c["y"], z=c["z"],
            rho=rr.rho, var1_r2=rr.var1_r2, n_eig=rr.n_eigenvalues_used,
            tau=tr.tau_integral, ts_var=var, ts_rms=rms,
            coreg_dist_mm=coreg_dist, **spec, **band_rho,
        ))
    df = pd.DataFrame(rows)
    df.attrs["elapsed_s"] = round(time.time() - t0, 1)
    df.attrs["coreg_dist_mm"] = coreg_dist
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1, help="max subjects to process")
    ap.add_argument("--subject", type=str, default=None, help="specific CCID e.g. CC110037")
    ap.add_argument("--no-tsss", action="store_true")
    ap.add_argument("--ica", action="store_true")
    ap.add_argument("--coreg", choices=["auto", "template"], default="auto",
                    help="auto = per-subject ICP coreg; template = generic fsaverage trans")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a subject's output CSV already exists")
    ap.add_argument("--task", choices=["rest", "smt", "passive"], default="rest",
                    help="which BIDS task recordings to process")
    ap.add_argument("--bands", action="store_true",
                    help="also compute per-band rho (delta/theta/alpha/beta/gamma)")
    ap.add_argument("--out", type=str, default="/home/salardini/data/camcan/derivatives")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading Schaefer-400 atlas + centroids ...")
    labels, centroids = load_schaefer400(SUBJECTS_DIR)
    print(f"  {len(labels)} parcels")

    # find complete subjects already on disk (full-size .fif)
    raw_root = os.path.join(BIDS_BASE, args.task)
    fifs = sorted(glob.glob(os.path.join(raw_root, "sub-*", "meg", f"*_task-{args.task}_meg.fif")))
    fifs = [f for f in fifs if os.path.getsize(f) > 50_000_000]
    if args.subject:
        fifs = [f for f in fifs if args.subject in f]
    fifs = fifs[: args.limit]
    print(f"processing {len(fifs)} subject(s); tSSS={not args.no_tsss} "
          f"ICA={args.ica} coreg={args.coreg}")

    all_df = []
    for f in fifs:
        subj = os.path.basename(f).split("_")[0]
        out = os.path.join(args.out, f"{subj}_rho_schaefer400.csv")
        if (not args.force) and os.path.exists(out) and os.path.getsize(out) > 1000:
            all_df.append(pd.read_csv(out))
            print(f"  {subj}: skip (exists)")
            continue
        try:
            df = process_subject(f, labels, centroids, PARAMS,
                                  do_tsss=not args.no_tsss, do_ica=args.ica,
                                  coreg_mode=args.coreg, do_bands=args.bands)
            out = os.path.join(args.out, f"{subj}_rho_schaefer400.csv")
            df.to_csv(out, index=False)
            from scipy.stats import spearmanr
            valid = df.dropna(subset=["rho", "z"])
            rs, pv = spearmanr(valid["rho"], valid["z"])
            cd = df.attrs.get("coreg_dist_mm", float("nan"))
            print(f"  {subj}: rho mean={df['rho'].mean():.3f} | "
                  f"DV Spearman(rho,z)={rs:+.3f} (p={pv:.1e}) | "
                  f"coreg_dist={cd:.1f}mm | {df.attrs['elapsed_s']}s")
            all_df.append(df)
        except Exception as e:
            import traceback
            print(f"  {subj}: FAILED {type(e).__name__}: {e}")
            traceback.print_exc()

    if all_df:
        combined = pd.concat(all_df, ignore_index=True)
        combined.to_csv(os.path.join(args.out, "rho_schaefer400_all.csv"), index=False)
        print(f"wrote combined: {len(all_df)} subject(s), {len(combined)} rows")


if __name__ == "__main__":
    main()
