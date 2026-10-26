"""Figures use saved CSV tables, never manually entered performance values."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
LABELS={'majority':'Majority','logistic_A':'Logistic A','logistic_B':'Logistic B','random_forest_B':'Random forest B','hist_gradient_B':'Gradient boosting B'}
def plot_rows(path,split):
    df=pd.read_csv(path);return df[(df.split==split)&df.policy.isin(['validation_f1','majority'])].copy()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();r=ROOT/'results'/a.run_id;out=ROOT/'figures'/a.run_id;out.mkdir(parents=True,exist_ok=True)
    split='test' if (r/'test_predictions.csv.gz').exists() else 'validation';m=plot_rows(r/'metrics.csv',split)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    def save(fig,name):fig.savefig(out/(name+'.png'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5));x=np.arange(len(m));w=.24
    for i,col in enumerate(['precision','recall','f1']):ax.bar(x+(i-1)*w,m[col],w,label=col.capitalize())
    ax.set(xticks=x,xticklabels=[LABELS[s] for s in m.model],ylim=(0,1),ylabel='Score',title=f'{a.run_id}: {split} performance')
    ax.legend();fig.text(.12,-.01,'2024 validation F1 thresholds; majority always slight. Majority precision is undefined (no positive predictions).',fontsize=8);save(fig,'model_performance')
    pr=pd.read_csv(r/'pr_curves.csv.gz');fig,ax=plt.subplots(figsize=(7,5))
    for name in m.model:
        q=pr[(pr.split==split)&pr.model.eq(name)];ap=m.set_index('model').loc[name,'average_precision']
        if name=='majority':ax.axhline(ap,linestyle='--',label=f'Majority AP={ap:.3f}')
        else:ax.step(q.recall,q.precision,where='post',label=f'{LABELS[name]} AP={ap:.3f}')
    ax.set(xlabel='Recall (Fatal / Serious)',ylabel='Precision',xlim=(0,1),ylim=(0,1),title=f'{split.capitalize()} precision–recall curves');ax.legend(fontsize=9);save(fig,'precision_recall')
    cal=pd.read_csv(r/'calibration.csv');fig,ax=plt.subplots(figsize=(7,5))
    for name in m.model:
        q=cal[(cal.split==split)&cal.model.eq(name)&cal.n.gt(0)];ax.plot(q.mean_probability,q.severe_fraction,'o-',label=LABELS[name])
    ax.plot([0,1],[0,1],'k--',alpha=.5);ax.set(xlabel='Mean predicted probability',ylabel='Observed severe fraction',xlim=(0,1),ylim=(0,1),title='Reliability plot (10 fixed-width bins; sparse bins retained)');ax.legend(fontsize=9);save(fig,'calibration')
    if (r/'permutation_importance.csv').exists():
        imp=pd.read_csv(r/'permutation_importance.csv').groupby(['model','feature']).ap_drop.agg(['mean','std']).reset_index();imp.to_csv(r/'importance_summary.csv',index=False)
        fig,axs=plt.subplots(1,3,figsize=(14,5))
        for ax,(name,g) in zip(axs,imp.groupby('model')):
            g=g.sort_values('mean');ax.barh(g.feature,g['mean'],xerr=g['std'],color='#287c8e');ax.axvline(0,color='black',lw=.6);ax.set(title=LABELS[name],xlabel='Validation AP decrease')
        fig.suptitle('Permutation importance: 5,000 validation records, 3 repeats (SD bars)');fig.tight_layout();save(fig,'feature_importance')
    if split=='test':
        dist=pd.read_csv(ROOT/'results/data/distributions.csv');fig,axs=plt.subplots(1,3,figsize=(12,4))
        for ax,feature in zip(axs,['collision_severity','speed_limit','light_conditions']):
            q=dist[dist.feature.eq(feature)].pivot(index='value',columns='year',values='fraction').fillna(0);q.plot.bar(ax=ax)
            labels={'collision_severity':{'1':'Fatal','2':'Serious','3':'Slight'},'light_conditions':{'1':'Daylight','4':'Dark: lit','5':'Dark: unlit','6':'Dark: no lights','7':'Dark: unknown','__MISSING__':'Missing'}}
            ax.set_xticklabels([labels.get(feature,{}).get(str(v),'Missing' if str(v)=='__MISSING__' else str(v)) for v in q.index],rotation=45,ha='right')
            ax.set(title=feature,xlabel='mph' if feature=='speed_limit' else '',ylabel='Fraction')
        fig.tight_layout();save(fig,'distributions')
        counts=pd.read_csv(ROOT/'results/data/sample_counts.csv').set_index('year');choices=json.loads((r/'frozen_choices.json').read_text());n=choices['models'][0]
        fig,ax=plt.subplots(figsize=(10,5));ax.axis('off')
        texts=[f"Official collision tables + dictionary\n2024: {counts.loc[2024,'eligible_rows']:,} | 2025: {counts.loc[2025,'eligible_rows']:,}",f"Dictionary-based cleaning and predictor allowlist\nTarget: Fatal / Serious = 1; Slight = 0",f"2024 stratified split (seed 20261021)\nTrain {n['train_n']:,} | Validation {n['validation_n']:,}",f"Train-only preprocessing + fixed models\nFreeze thresholds on 2024 validation",f"One 2025 temporal test: {counts.loc[2025,'eligible_rows']:,} collisions\nMetrics, PR curves, calibration; no test tuning"]
        for i,t in enumerate(texts):
            y=.94-i*.2;ax.text(.5,y,t,ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',facecolor='#e3f0f4',edgecolor='#287c8e'))
            if i<4:ax.annotate('',xy=(.5,y-.14),xytext=(.5,y-.06),arrowprops=dict(arrowstyle='->'))
        save(fig,'workflow')
    files=[r/'metrics.csv',r/'pr_curves.csv.gz',r/'calibration.csv']
    if split=='test':files+=[r/'permutation_importance.csv',ROOT/'results/data/distributions.csv',ROOT/'results/data/sample_counts.csv']
    (out/'inputs.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
if __name__=='__main__':main()
