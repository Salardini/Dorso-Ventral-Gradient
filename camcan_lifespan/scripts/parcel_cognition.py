#!/usr/bin/env python
"""parcel_cognition.py — PARCEL-WISE brain-cognition maps (not whole-brain summaries).

For each of the 400 Schaefer parcels independently, partial-correlate that parcel's
rho (and tau) across subjects with each cognitive domain, controlling age + sex.
Produces a 400-value map per (metric, domain), FDR-corrected across parcels, and
localizes the significant parcels by Yeo-7 network + hemisphere with their sign.

Run for the whole cohort (age regressed) and the elderly (60-88, where the global
analysis showed the rho-cognition coupling actually lives).
"""
import pandas as pd, numpy as np, glob
from statsmodels.stats.multitest import multipletests
from scipy import stats
ROOT="/home/salardini/data/camcan"
COG={"ACER":"additional_acer","memory":"additional_memory","fluency":"additional_fluencies",
     "story_imm":"homeint_storyrecall_i","story_del":"homeint_storyrecall_d","STW":"additional_STW_total"}

ap=pd.read_csv(f"{ROOT}/raw/dataman_02034/approved_data.tsv",sep="\t")
idc=[c for c in ap.columns if c.lower()=="ccid"][0]
cog=ap[[idc]+list(COG.values())].rename(columns={idc:"subject",**{v:k for k,v in COG.items()}})
std=pd.read_csv(f"{ROOT}/raw/dataman_02034/standard_data.csv")
age=std.set_index("CCID").Age; sex=std.set_index("CCID").Sex

# build subject x parcel matrices
frames=[pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/derivatives_final/sub-*_rho_schaefer400.csv"))]
labels=frames[0].sort_values("parcel_idx").label.values
net=np.array([l.split("_")[2] for l in labels]); hemi=np.array([l.split("_")[1] for l in labels])
def mat(metric):
    rows={f.subject.iloc[0]: f.sort_values("parcel_idx")[metric].values for f in frames}
    return pd.DataFrame.from_dict(rows, orient="index")   # subjects x 400
RHO=mat("rho"); TAU=mat("tau")
meta=pd.DataFrame({"subject":RHO.index}); meta["age"]=meta.subject.map(age); meta["sex"]=meta.subject.map(sex)
meta=meta.merge(cog,on="subject",how="left").set_index("subject")

def partial_map(M, y, covars):
    """partial corr of each column of M with y, controlling covars. returns r,p arrays."""
    keep=y.notna() & covars.notna().all(axis=1)
    M=M.loc[keep].values; y=y.loc[keep].values
    D=np.column_stack([np.ones(keep.sum()), covars.loc[keep].values])
    def resid(A): return A - D@np.linalg.lstsq(D,A,rcond=None)[0]
    rM=resid(M); ry=resid(y.reshape(-1,1))[:,0]
    rM=(rM-rM.mean(0))/ (rM.std(0)+1e-12); ry=(ry-ry.mean())/(ry.std()+1e-12)
    n=keep.sum(); r=(rM*ry[:,None]).mean(0)
    df=n-D.shape[1]-1; t=r*np.sqrt(df/np.clip(1-r**2,1e-12,None))
    p=2*stats.t.sf(np.abs(t),df); return r,p,n

def run(mask,label,metrics):
    print(f"\n################ {label}  (n_subjects={mask.sum()}) ################")
    sub=meta[mask]
    for mname,Mfull in metrics:
        M=Mfull.loc[sub.index]
        for dom in COG:
            covars=sub[["age"]].assign(sex=(sub.sex=="MALE").astype(float))
            r,p,n=partial_map(M, sub[dom], covars)
            q=multipletests(p,method="fdr_bh")[1]
            sig=q<0.05
            if sig.sum()==0:
                print(f"  {mname}->{dom:10s}: 0 parcels (min q={q.min():.3f})"); continue
            # network breakdown + direction
            nets=pd.Series(net[sig]).value_counts()
            pos=(r[sig]>0).sum(); neg=(r[sig]<0).sum()
            top=", ".join(f"{k}:{v}" for k,v in nets.head(4).items())
            print(f"  {mname}->{dom:10s}: {sig.sum():3d} parcels FDR  (+{pos}/-{neg})  nets[{top}]  |r|max={np.abs(r[sig]).max():.2f}")

run(meta.age.notna(), "WHOLE COHORT (age+sex regressed)", [("rho",RHO),("tau",TAU)])
run((meta.age>=60), "ELDERLY 60-88", [("rho",RHO),("tau",TAU)])
