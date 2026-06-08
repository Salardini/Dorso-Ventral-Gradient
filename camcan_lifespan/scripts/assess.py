#!/usr/bin/env python
"""assess.py <derivatives_dir> — evaluate a CamCAN rho run against the published
MOUS gradient. Prints per-subject DV gradient, the group rho map's DV gradient,
and the decisive Spearman(published_rho_map, camcan_rho_map)."""
import sys, glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

PUB = "/home/salardini/projects/Dorso-Ventral-Gradient/data/revision/parcel_group_maps.csv"

def main(deriv):
    fs = sorted(glob.glob(os.path.join(deriv, "sub-*_rho_schaefer400.csv")))
    if not fs:
        print(f"no subject CSVs in {deriv}"); return
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    n = df.subject.nunique()
    print(f"== {deriv} :: {n} subjects, {len(df)} rows ==")

    if "coreg_dist_mm" in df:
        cd = df.groupby("subject").coreg_dist_mm.first()
        print(f"coreg_dist_mm: median {cd.median():.1f} (range {cd.min():.1f}-{cd.max():.1f})")

    per = np.array([spearmanr(g.rho, g.z)[0] for _, g in df.groupby("subject")])
    print(f"per-subject Spearman(rho,z): mean {per.mean():+.3f} sd {per.std():.3f} "
          f"| {(per<0).sum()}/{len(per)} negative")

    g = df.groupby("label").agg(rho=("rho","mean"), z=("z","first"),
                                y=("y","first")).reset_index()
    print(f"GROUP Spearman(rho,z) = {spearmanr(g.rho,g.z)[0]:+.3f} "
          f"(AP: {spearmanr(g.rho,g.y)[0]:+.3f}) | published target -0.73")
    print(f"group rho spread: {g.rho.max()-g.rho.min():.3f}")

    pub = pd.read_csv(PUB)[["label","rho_mean","z"]]
    m = pub.merge(g, on="label")
    r_map = spearmanr(m.rho_mean, m.rho)[0]
    print(f"*** Spearman(published_map, camcan_map) = {r_map:+.3f}  "
          f"({'RECOVERED' if r_map>0.4 else 'still off'}) ***")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/salardini/data/camcan/derivatives")
