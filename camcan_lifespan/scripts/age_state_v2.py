#!/usr/bin/env python
"""age_state_v2.py — adds (a) bootstrap CI bands on the lifespan curves,
(b) non-linear (quadratic) age model + plateau test, (c) sex-split curves."""
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
rng=np.random.RandomState(0)

def load_matrix(deriv):
    """returns R[n_subj x n_parcel] rho, z[n_parcel], pub[n_parcel], meta df."""
    df=pd.concat([pd.read_csv(f) for f in
                  sorted(glob.glob(f"{ROOT}/{deriv}/sub-*_rho_schaefer400.csv"))],
                 ignore_index=True)
    df["age"]=df.subject.map(age); df["sex"]=df.subject.map(sex)
    df=df.dropna(subset=["age"])
    piv=df.pivot_table(index="subject",columns="label",values="rho")
    labels=piv.columns
    z=df.groupby("label").z.first().reindex(labels).values
    pub=pubg.reindex(labels).values
    meta=df.groupby("subject").agg(age=("age","first"),sex=("sex","first")).reindex(piv.index)
    return piv.values, z, pub, meta

def gdv(R_rows, z):              # group DV slope from a set of subject rows
    return spearmanr(R_rows.mean(0), z)[0]

conds={"rest":"derivatives_final","passive":"derivatives_passive"}
M={k:load_matrix(v) for k,v in conds.items()}

# ---------- (a) bootstrap CI sliding-window curves ----------
centers=np.arange(22,82,3); half=6; B=1000
def curve_ci(R,z,ages,sexmask=None):
    xs,med,lo,hi=[],[],[],[]
    for c in centers:
        m=(ages>=c-half)&(ages<=c+half)
        if sexmask is not None: m=m&sexmask
        idx=np.where(m)[0]
        if len(idx)<15: continue
        boots=[gdv(R[rng.choice(idx,len(idx),replace=True)],z) for _ in range(B)]
        xs.append(c); med.append(gdv(R[idx],z))
        lo.append(np.percentile(boots,2.5)); hi.append(np.percentile(boots,97.5))
    return np.array(xs),np.array(med),np.array(lo),np.array(hi)

colors={"rest":"#1f77b4","passive":"#d62728"}
fig,ax=plt.subplots(figsize=(8.4,5.4))
for k,(R,z,pub,meta) in M.items():
    x,m,lo,hi=curve_ci(R,z,meta.age.values)
    ax.fill_between(x,lo,hi,color=colors[k],alpha=0.18,lw=0)
    ax.plot(x,m,"-o",color=colors[k],lw=2,ms=4,label=k)
ax.axhline(-0.731,ls="--",color="gray",lw=1.2); ax.text(70,-0.731+0.02,"MOUS rest -0.73",color="gray",fontsize=8)
ax.axhline(0,color="k",lw=0.6)
ax.set_xlabel("age (centre of ±6-yr window)"); ax.set_ylabel("group DV slope  Spearman(ρ,z)")
ax.set_title("DV ρ-gradient across the lifespan (95% bootstrap CI)")
ax.legend(title="MEG state"); ax.invert_yaxis(); fig.tight_layout()
fig.savefig(f"{ROOT}/writeup/lifespan_dv_gradient_CI.png",dpi=140)
print("saved writeup/lifespan_dv_gradient_CI.png")

# ---------- (c) sex-split figure ----------
fig2,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
sexcol={"MALE":"#2c7fb8","FEMALE":"#d95f0e"}
for ax2,(k,(R,z,pub,meta)) in zip(axes,M.items()):
    for sx in ["MALE","FEMALE"]:
        mask=(meta.sex.values==sx)
        x,m,lo,hi=curve_ci(R,z,meta.age.values,sexmask=mask)
        ax2.fill_between(x,lo,hi,color=sexcol[sx],alpha=0.15,lw=0)
        ax2.plot(x,m,"-o",color=sexcol[sx],lw=1.8,ms=3,label=sx.title())
    ax2.axhline(-0.731,ls="--",color="gray",lw=1); ax2.axhline(0,color="k",lw=0.6)
    ax2.set_title(k); ax2.set_xlabel("age"); ax2.invert_yaxis(); ax2.legend()
axes[0].set_ylabel("group DV slope  Spearman(ρ,z)")
fig2.suptitle("DV ρ-gradient lifespan by sex"); fig2.tight_layout()
fig2.savefig(f"{ROOT}/writeup/lifespan_dv_gradient_sex.png",dpi=140)
print("saved writeup/lifespan_dv_gradient_sex.png")

# ---------- (b) non-linear age model ----------
def per_subject_long():
    rows=[]
    for k,(R,z,pub,meta) in M.items():
        sl=np.array([spearmanr(R[i],z)[0] for i in range(R.shape[0])])
        for i,s in enumerate(meta.index):
            rows.append((s,sl[i],meta.age.values[i],meta.sex.values[i],k))
    d=pd.DataFrame(rows,columns=["subject","dv","age","sex","state"])
    d["age_c"]=d.age-d.age.mean(); d["age_c2"]=d.age_c**2
    d["state"]=pd.Categorical(d.state,["rest","passive"]); d["sex"]=d.sex.astype("category")
    return d
d=per_subject_long()
lin=smf.mixedlm("dv ~ age_c*C(state) + C(sex)", d, groups=d.subject).fit(method="lbfgs",reml=False)
quad=smf.mixedlm("dv ~ (age_c+age_c2)*C(state) + C(sex)", d, groups=d.subject).fit(method="lbfgs",reml=False)
print("\n========== LINEAR vs QUADRATIC age (ML, lower AIC better) ==========")
print(f"  linear    AIC={lin.aic:.1f}")
print(f"  quadratic AIC={quad.aic:.1f}   dAIC={lin.aic-quad.aic:+.1f}")
print("\n  quadratic model terms:")
for t in quad.params.index:
    if t=="Group Var": continue
    print(f"    {t:34s} beta={quad.params[t]:+.5f}  p={quad.pvalues[t]:.2e}")
# where does rest curve flatten / turn? vertex of quadratic for rest
b=quad.params; amean=d.age.mean()
a1,a2=b["age_c"],b["age_c2"]
vertex=amean - a1/(2*a2) if a2!=0 else float("nan")
print(f"\n  rest quadratic vertex (slope-flattening age) ~ {vertex:.0f} yr")
print(f"  age2 main p={quad.pvalues['age_c2']:.2e} | age2xstate p={quad.pvalues['age_c2:C(state)[T.passive]']:.2e}")
