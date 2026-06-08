#!/usr/bin/env python
"""network_cognition.py — Yeo-7 network x cognitive-domain partial correlations for
rho AND tau (age+sex adjusted, BH-FDR over the 7x6 grid per metric). Prints both
tables and renders a compact two-panel heatmap.
"""
import pandas as pd, numpy as np, glob
from statsmodels.stats.multitest import multipletests
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/home/salardini/data/camcan"
COG={"ACER":"additional_acer","memory":"additional_memory","fluency":"additional_fluencies",
     "story_imm":"homeint_storyrecall_i","story_del":"homeint_storyrecall_d","STW":"additional_STW_total"}
NETS=["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
ap=pd.read_csv(f"{ROOT}/raw/dataman_02034/approved_data.tsv",sep="\t"); idc=[c for c in ap.columns if c.lower()=="ccid"][0]
cog=ap[[idc]+list(COG.values())].rename(columns={idc:"subject",**{v:k for k,v in COG.items()}})
std=pd.read_csv(f"{ROOT}/raw/dataman_02034/standard_data.csv"); age=std.set_index("CCID").Age; sex=std.set_index("CCID").Sex
frames=[pd.read_csv(f) for f in sorted(glob.glob(f"{ROOT}/derivatives_final/sub-*_rho_schaefer400.csv"))]

def netmat(metric):
    rows={}
    for f in frames:
        g=f.copy(); g["net"]=g.label.str.split("_").str[2]
        rows[f.subject.iloc[0]]=g.groupby("net")[metric].mean()
    return pd.DataFrame(rows).T
m=pd.DataFrame({"subject":netmat("rho").index})
m["age"]=m.subject.map(age); m["sex"]=(m.subject.map(sex)=="MALE").astype(float)
m=m.merge(cog,on="subject",how="left").set_index("subject")
D=m[[]].assign(c=1,age=m.age,sex=m.sex)[["c","age","sex"]]

def partial(x,y):
    keep=x.notna()&y.notna(); x=x[keep].values.astype(float); y=y[keep].values.astype(float); d=D[keep].values
    R=lambda A:A-d@np.linalg.lstsq(d,A,rcond=None)[0]
    rx=R(x.reshape(-1,1))[:,0]; ry=R(y.reshape(-1,1))[:,0]
    r=np.corrcoef(rx,ry)[0,1]; df=keep.sum()-d.shape[1]-1
    return r,2*stats.t.sf(abs(r*np.sqrt(df/max(1-r**2,1e-12))),df)

fig,axes=plt.subplots(1,2,figsize=(13,4.6))
for ax,metric in zip(axes,["rho","tau"]):
    M=netmat(metric).loc[m.index]
    Rm=np.zeros((len(NETS),len(COG))); Pm=np.zeros_like(Rm)
    for i,net in enumerate(NETS):
        for j,k in enumerate(COG):
            Rm[i,j],Pm[i,j]=partial(M[net],m[k])
    q=multipletests(Pm.ravel(),method="fdr_bh")[1].reshape(Pm.shape)
    print(f"\n=== {metric.upper()} network x domain (partial r, * FDR q<.05) ===")
    print(f"{'net':12s}"+"".join(f"{k:>11s}" for k in COG))
    for i,net in enumerate(NETS):
        print(f"{net:12s}"+"".join(f"{Rm[i,j]:>+9.2f}{'*' if q[i,j]<.05 else ' '} " for j in range(len(COG))))
    im=ax.imshow(Rm,cmap="RdBu_r",vmin=-0.18,vmax=0.18,aspect="auto")
    ax.set_xticks(range(len(COG))); ax.set_xticklabels(COG,rotation=40,ha="right",fontsize=9)
    ax.set_yticks(range(len(NETS))); ax.set_yticklabels(NETS,fontsize=9)
    ax.set_title(f"{metric}  (Yeo-7 mean) -> cognition\nage+sex adj, * = FDR q<.05")
    for i in range(len(NETS)):
        for j in range(len(COG)):
            ax.text(j,i,f"{Rm[i,j]:+.2f}"+("*" if q[i,j]<.05 else ""),ha="center",va="center",
                    fontsize=7,color="black" if abs(Rm[i,j])<0.12 else "white")
    fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,label="partial r")
fig.suptitle("CamCAN resting MEG: network dynamics -> cognition (n=646, age+sex controlled)",fontsize=11)
fig.tight_layout()
out=f"{ROOT}/writeup/network_cognition.png"; fig.savefig(out,dpi=140); print("\nsaved",out)
