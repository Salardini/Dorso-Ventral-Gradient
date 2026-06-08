#!/usr/bin/env python
"""age_state_analysis.py — lifespan curve of the DV rho-gradient (rest vs passive)
and a formal age x state interaction model with sex covariate.

Outputs:
  derivatives_final/../writeup/lifespan_dv_gradient.png
  prints mixed-model + sliding-window stats
"""
import glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT="/home/salardini/data/camcan"
PUB="/home/salardini/projects/Dorso-Ventral-Gradient/data/revision/parcel_group_maps.csv"
pubg=pd.read_csv(PUB).set_index("label")["rho_mean"]
demo=pd.read_csv(f"{ROOT}/raw/dataman_02034/standard_data.csv")
age=demo.set_index("CCID")["Age"]; sex=demo.set_index("CCID")["Sex"]

def load(deriv):
    df=pd.concat([pd.read_csv(f) for f in
                  sorted(glob.glob(f"{ROOT}/{deriv}/sub-*_rho_schaefer400.csv"))],
                 ignore_index=True)
    df["age"]=df.subject.map(age); df["sex"]=df.subject.map(sex)
    return df.dropna(subset=["age"])

def group_dv(df):
    """group-mean rho map -> Spearman(rho,z) and map_r vs published."""
    g=df.groupby("label").agg(rho=("rho","mean"),z=("z","first")).reset_index()
    m=g.set_index("label").join(pubg.rename("pub"),how="inner").dropna()
    return spearmanr(g.rho,g.z)[0], spearmanr(m.rho,m.pub)[0]

def per_subject(df):
    """per-subject DV slope + age + sex, one row per subject."""
    rows=[]
    for s,g in df.groupby("subject"):
        rows.append((s, spearmanr(g.rho,g.z)[0], g.age.iloc[0], g.sex.iloc[0]))
    return pd.DataFrame(rows,columns=["subject","dv","age","sex"])

conds={"rest":"derivatives_final","passive":"derivatives_passive"}
data={k:load(v) for k,v in conds.items()}
ps={k:per_subject(v) for k,v in data.items()}

# ---- sliding-window GROUP DV slope vs age ----
centers=np.arange(22,82,3); half=6  # +/-6yr window
fig,ax=plt.subplots(figsize=(8,5.2))
colors={"rest":"#1f77b4","passive":"#d62728"}
curves={}
for k,df in data.items():
    xs,ys,ns=[],[],[]
    for c in centers:
        w=df[(df.age>=c-half)&(df.age<=c+half)]
        if w.subject.nunique()<15: continue
        dv,_=group_dv(w); xs.append(c); ys.append(dv); ns.append(w.subject.nunique())
    curves[k]=(xs,ys,ns)
    ax.plot(xs,ys,"-o",color=colors[k],lw=2,ms=4,label=f"{k}")
ax.axhline(-0.731,ls="--",color="gray",lw=1.2)
ax.text(72,-0.731+0.02,"published MOUS rest -0.73",color="gray",fontsize=8)
ax.axhline(0,color="k",lw=0.6)
ax.set_xlabel("age (centre of ±6-yr window)"); ax.set_ylabel("group DV slope  Spearman(ρ, z)")
ax.set_title("Dorsoventral ρ-gradient across the lifespan (CamCAN)")
ax.legend(title="MEG state"); ax.invert_yaxis(); fig.tight_layout()
out=f"{ROOT}/writeup/lifespan_dv_gradient.png"; fig.savefig(out,dpi=140)
print("saved",out)

# ---- formal age x state interaction (per-subject, paired across conditions) ----
long=pd.concat([ps["rest"].assign(state="rest"),
                ps["passive"].assign(state="passive")],ignore_index=True)
long["age_c"]=long.age-long.age.mean()
long["state"]=pd.Categorical(long.state,categories=["rest","passive"])
long["sex"]=long.sex.astype("category")
print("\nN obs:",len(long),"| subjects rest",ps['rest'].shape[0],"passive",ps['passive'].shape[0])

# mixed model: random intercept per subject (paired rest/passive)
md=smf.mixedlm("dv ~ age_c * C(state) + C(sex)", long, groups=long["subject"])
mf=md.fit(method="lbfgs", reml=True)
print("\n========== MIXED MODEL: dv ~ age_c * state + sex ==========")
for term in mf.params.index:
    if term in ("Group Var",): continue
    print(f"  {term:32s} beta={mf.params[term]:+.4f}  p={mf.pvalues[term]:.2e}")
print("  (age_c:C(state)[T.passive] is the AGE x STATE interaction)")

# slopes per state from the model
b=mf.params
print("\n  rest    DV-vs-age slope = %+.4f /yr"%b["age_c"])
print("  passive DV-vs-age slope = %+.4f /yr"%(b["age_c"]+b["age_c:C(state)[T.passive]"]))

# sliding-window numeric table
print("\n========== sliding-window group DV slope (n>=15) ==========")
print("%5s | %18s | %18s"%("age","rest dv (n)","passive dv (n)"))
allc=sorted(set(curves["rest"][0])|set(curves["passive"][0]))
for c in allc:
    r=dict(zip(curves["rest"][0],zip(curves["rest"][1],curves["rest"][2]))).get(c)
    p=dict(zip(curves["passive"][0],zip(curves["passive"][1],curves["passive"][2]))).get(c)
    rs=f"{r[0]:+.3f} ({r[1]})" if r else "--"
    psx=f"{p[0]:+.3f} ({p[1]})" if p else "--"
    print("%5d | %18s | %18s"%(c,rs,psx))
