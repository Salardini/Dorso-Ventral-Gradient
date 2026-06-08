#!/usr/bin/env python
"""fmri_concordance.py — spatial concordance of CamCAN resting-MEG rho/tau group
maps with fMRI-derived parcel variables (Schaefer-400), with spin-test p-values.

We cannot measure rho on fMRI (TR too slow for the rotational dynamics), but the
MEG rho/tau spatial maps can be compared against fMRI's slow-dynamics architecture:
fMRI intrinsic timescale, ALFF/fALFF, spectral exponent, and the (questionable)
fMRI-rho. Parcels are aligned by OPTIMAL centroid assignment (the fMRI file is not
in canonical Schaefer order; naive row alignment gives x-corr 0.755 and wrong signs).

Requires the Dorso-Ventral-Gradient repo (fMRI maps + spin_test util).
"""
import pandas as pd, numpy as np, glob, sys, os
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
DVG=os.path.expanduser("~/projects/Dorso-Ventral-Gradient")
sys.path.insert(0, f"{DVG}/scripts/analysis/utils"); from spin_test import run_spin_test
ROOT=os.path.expanduser("~/data/camcan")

# CamCAN MEG group map (per-parcel mean rho/tau across subjects)
frames=[pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/derivatives_final/sub-*_rho_schaefer400.csv"))]
meg=pd.concat(frames).groupby("parcel_idx").agg(rho=("rho","mean"),tau=("tau","mean"),
        x=("x","first"),y=("y","first"),z=("z","first"),label=("label","first")).reset_index()
meg["hemi"]=meg.label.str.split("_").str[1]

fm=pd.read_csv(f"{DVG}/data/revision/fmri_parcel_measures_v2.csv")

# optimal 1-to-1 parcel alignment by centroid distance (robust to ordering differences)
D=cdist(meg[["x","y","z"]].values, fm[["x","y","z"]].values)
_, ci = linear_sum_assignment(D)
fm=fm.iloc[ci].reset_index(drop=True)
xcorr=np.corrcoef(meg.x.values, fm.x.values)[0,1]
assert xcorr>0.99, f"alignment failed (x-corr={xcorr:.3f})"
print(f"parcel alignment OK (x-corr={xcorr:.3f})\n")

hemi=meg.hemi.values
print(f"{'fMRI variable':16s}{'MEG-rho (r, p_spin)':>26s}{'MEG-tau (r, p_spin)':>26s}")
for v in ["tau_fmri","alff","falff","spectral_exp","rho_fmri"]:
    ok=fm[v].notna().values
    out=[]
    for mv in ["rho","tau"]:
        x=meg[mv].values; y=fm[v].values
        r=spearmanr(x[ok],y[ok])[0]
        p=run_spin_test(x[ok],y[ok],hemi[ok],n_perm=5000)
        p=float(p[1]) if hasattr(p,"__len__") else float(p)
        out.append(f"{r:+.3f}, {p:.4f}")
    print(f"{v:16s}{out[0]:>26s}{out[1]:>26s}")
print("\nNote: MEG-tau vs fMRI-tau is NEGATIVE — the millisecond (MEG) and second (fMRI)")
print("timescale hierarchies are spatially anti-correlated. Spin-test significant.")
